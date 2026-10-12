"""Streaming forecast-verification metrics over ocean pixels, per lead time.

All inputs are physical units (m/s) with shape (B, L, 2, H, W), where L is the
number of lead times. ``mask`` is (H, W) boolean, True over ocean.
"""
from __future__ import annotations

import math

import torch

# Fraction of a standard normal within +/- k sigma
GAUSSIAN_COVERAGE = {1: math.erf(1 / math.sqrt(2)), 2: math.erf(2 / math.sqrt(2))}


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma^2) for observation y.

    Reduces to |y - mu| as sigma -> 0, so it is directly comparable with the MAE
    of a deterministic forecast. Lower is better.
    """
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


class LeadTimeMetrics:
    """Accumulates sums over batches so the full test set never sits in memory."""

    def __init__(self, n_leads: int, mask: torch.Tensor):
        self.mask = mask.bool()
        self.n_px = int(self.mask.sum())
        self.count = 0
        z = lambda: torch.zeros(n_leads, dtype=torch.float64)  # noqa: E731
        self.sq_vec, self.abs_comp = z(), z()
        self.crps, self.sigma_sq = z(), z()
        self.cov = {k: z() for k in GAUSSIAN_COVERAGE}
        self.probabilistic = False

    def _ocean(self, x):
        # (B, L, 2, H, W) -> (B, L, 2, n_ocean)
        return x[..., self.mask]

    def update(self, pred: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor | None = None):
        p, y = self._ocean(pred).double(), self._ocean(target).double()
        err = p - y
        b = p.shape[0]
        self.count += b
        self.sq_vec += (err**2).sum(dim=2).mean(dim=-1).sum(dim=0).cpu()
        self.abs_comp += err.abs().mean(dim=(2, 3)).sum(dim=0).cpu()
        if sigma is not None:
            self.probabilistic = True
            s = self._ocean(sigma).double().clamp_min(1e-6)
            self.crps += gaussian_crps(p, s, y).mean(dim=(2, 3)).sum(dim=0).cpu()
            self.sigma_sq += (s**2).mean(dim=(2, 3)).sum(dim=0).cpu()
            for k in self.cov:
                self.cov[k] += (err.abs() <= k * s).double().mean(dim=(2, 3)).sum(dim=0).cpu()

    def result(self) -> dict:
        n = max(self.count, 1)
        out = {
            # RMS of the vector error |(du, dv)|: a typical current error in m/s
            "rmse_vector": torch.sqrt(self.sq_vec / n).tolist(),
            "mae_component": (self.abs_comp / n).tolist(),
            "n_samples": self.count,
            "n_ocean_pixels": self.n_px,
        }
        if self.probabilistic:
            out["crps"] = (self.crps / n).tolist()
            # Spread: RMS predicted sigma per component. Calibrated => spread ~ RMSE per component.
            out["spread_component"] = torch.sqrt(self.sigma_sq / n).tolist()
            out["rmse_component"] = (torch.sqrt(self.sq_vec / n) / math.sqrt(2)).tolist()
            for k in self.cov:
                out[f"coverage_{k}sigma"] = (self.cov[k] / n).tolist()
        return out


def skill(rmse_model, rmse_reference):
    """Skill score 1 - RMSE_model / RMSE_reference (positive = better than reference)."""
    return [1 - m / r for m, r in zip(rmse_model, rmse_reference, strict=True)]
