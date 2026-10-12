"""Loading CMEMS surface-current NetCDF files and building training windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _find_dim(ds, candidates):
    for c in candidates:
        if c in ds.dims or c in ds.coords:
            return c
    return None


def load_cmems_uv(
    path: str,
    u_var: str = "utotal",
    v_var: str = "vtotal",
    depth_index: int = 0,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    regrid_hw: tuple[int, int] | None = None,
    fillna: float | None = 0.0,
):
    """Load a CMEMS-style NetCDF into a ``(T, 2, H, W)`` float32 array of (u, v).

    Dimension names are auto-detected (``latitude``/``lat``, ``longitude``/``lon``,
    optional ``depth``). Optionally crops to a bounding box and bilinearly regrids
    to a fixed ``H x W``. Land pixels are NaN in the source and are replaced by
    ``fillna`` (pass ``None`` to keep them, e.g. to build an ocean mask).

    Returns ``(uv, time, lat, lon)``.
    """
    import xarray as xr  # heavy import, only needed when actually loading data

    ds = xr.open_dataset(path, decode_times=True)
    time_dim = _find_dim(ds, ["time", "Times"])
    lat_dim = _find_dim(ds, ["latitude", "lat", "y"])
    lon_dim = _find_dim(ds, ["longitude", "lon", "x"])
    depth_dim = _find_dim(ds, ["depth", "depthu", "depthv", "z"])
    if time_dim is None or lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not detect dims. Found dims={list(ds.dims)} coords={list(ds.coords)}")
    for name in (u_var, v_var):
        if name not in ds.data_vars:
            raise KeyError(f"Variable '{name}' not found in dataset vars: {list(ds.data_vars)}")

    u, v = ds[u_var], ds[v_var]
    if depth_dim is not None and depth_dim in u.dims:
        u = u.isel({depth_dim: depth_index})
        v = v.isel({depth_dim: depth_index})

    if lat_min is not None or lat_max is not None:
        sl = slice(lat_min if lat_min is not None else float(ds[lat_dim].min()),
                   lat_max if lat_max is not None else float(ds[lat_dim].max()))
        u, v = u.sel({lat_dim: sl}), v.sel({lat_dim: sl})
    if lon_min is not None or lon_max is not None:
        sl = slice(lon_min if lon_min is not None else float(ds[lon_dim].min()),
                   lon_max if lon_max is not None else float(ds[lon_dim].max()))
        u, v = u.sel({lon_dim: sl}), v.sel({lon_dim: sl})

    if regrid_hw is not None:
        h, w = regrid_hw
        lat_new = np.linspace(float(u[lat_dim].min()), float(u[lat_dim].max()), h)
        lon_new = np.linspace(float(u[lon_dim].min()), float(u[lon_dim].max()), w)
        u = u.interp({lat_dim: lat_new, lon_dim: lon_new})
        v = v.interp({lat_dim: lat_new, lon_dim: lon_new})

    u = u.transpose(time_dim, lat_dim, lon_dim)
    v = v.transpose(time_dim, lat_dim, lon_dim)
    u_np, v_np = u.to_numpy(), v.to_numpy()
    if fillna is not None:
        u_np = np.nan_to_num(u_np, nan=fillna)
        v_np = np.nan_to_num(v_np, nan=fillna)

    uv = np.stack([u_np, v_np], axis=1).astype(np.float32)
    time, lat, lon = ds[time_dim].to_numpy(), u[lat_dim].to_numpy(), u[lon_dim].to_numpy()
    ds.close()
    return uv, time, lat, lon


# --------------------------------------------------------------------------
# Normalisation (statistics are always computed on the training period only)
# --------------------------------------------------------------------------
@dataclass
class ZScoreStats:
    u_mean: float
    u_std: float
    v_mean: float
    v_std: float

    @classmethod
    def fit(cls, uv: np.ndarray) -> ZScoreStats:
        u, v = uv[:, 0], uv[:, 1]
        return cls(float(np.mean(u)), float(np.std(u) + 1e-8), float(np.mean(v)), float(np.std(v) + 1e-8))

    @classmethod
    def from_dict(cls, d: dict) -> ZScoreStats:
        return cls(**{k: float(d[k]) for k in ("u_mean", "u_std", "v_mean", "v_std")})

    def to_dict(self) -> dict:
        return asdict(self)

    def _vec(self, xp_like, which: str):
        mean = [self.u_mean, self.v_mean]
        std = [self.u_std, self.v_std]
        vals = mean if which == "mean" else std
        if isinstance(xp_like, torch.Tensor):
            return torch.tensor(vals, dtype=xp_like.dtype, device=xp_like.device)
        return np.asarray(vals, dtype=xp_like.dtype)

    def _bcast(self, x, which):
        # Channel axis is the one of size 2 at position -3: (..., 2, H, W)
        return self._vec(x, which).reshape(2, 1, 1)

    def normalize(self, uv):
        """(..., 2, H, W) physical m/s -> z-scores."""
        return (uv - self._bcast(uv, "mean")) / self._bcast(uv, "std")

    def denormalize(self, uv_n):
        """(..., 2, H, W) z-scores -> physical m/s."""
        return uv_n * self._bcast(uv_n, "std") + self._bcast(uv_n, "mean")

    def denormalize_sigma(self, sigma_n):
        """Scale a predicted standard deviation from z-score units to m/s."""
        return sigma_n * self._bcast(sigma_n, "std")


def time_split(uv: np.ndarray, train_frac: float):
    """Chronological split: the first ``train_frac`` of frames train, the rest validate.

    Never shuffled across time, so no future information leaks into training.
    """
    split = int(len(uv) * train_frac)
    return uv[:split], uv[split:]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class SlidingWindowDataset(Dataset):
    """Sliding windows over a ``(T, 2, H, W)`` array.

    Item ``i`` is ``(X, Y)`` with ``X = uv[i : i+seq_len]`` (the input history)
    and ``Y = uv[i+seq_len : i+seq_len+horizon]`` (the next ``horizon`` frames).
    With ``horizon=1`` it serves one-step training; with ``horizon>1`` it serves
    autoregressive curriculum training and multi-step evaluation.
    """

    def __init__(self, uv: np.ndarray, seq_len: int = 12, horizon: int = 1):
        if uv.ndim != 4 or uv.shape[1] != 2:
            raise ValueError(f"uv must be (T, 2, H, W), got {uv.shape}")
        self.uv = uv
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.n = len(uv) - self.seq_len - self.horizon + 1
        if self.n <= 0:
            raise ValueError(f"Series too short: T={len(uv)} < seq_len + horizon = {self.seq_len + self.horizon}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        e = idx + self.seq_len
        x = torch.from_numpy(self.uv[idx:e]).float()
        y = torch.from_numpy(self.uv[e:e + self.horizon]).float()
        return x, y
