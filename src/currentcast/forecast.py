"""Autoregressive multi-step forecasting."""
from __future__ import annotations

import torch


def rollout(model, x_seq: torch.Tensor, steps: int, grad: bool = False):
    """Roll a one-step model forward ``steps`` times, feeding back its own mean.

    ``model(x)`` must return ``(mu, logvar)`` for ``x`` of shape (B, T, 2, H, W).
    The input window slides forward one frame per step: the oldest frame is
    dropped and the predicted mean appended.

    Returns ``(mu, logvar)``, each (B, steps, 2, H, W).
    """
    mus, logvars = [], []
    window = x_seq
    with torch.set_grad_enabled(grad):
        for _ in range(steps):
            mu, logvar = model(window)
            mus.append(mu)
            logvars.append(logvar)
            window = torch.cat([window[:, 1:], mu.unsqueeze(1)], dim=1)
    return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)


def persistence(x_seq: torch.Tensor, steps: int) -> torch.Tensor:
    """Naive baseline: the last observed frame, repeated for every lead time."""
    return x_seq[:, -1:].expand(-1, steps, -1, -1, -1)
