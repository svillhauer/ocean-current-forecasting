#!/usr/bin/env python3
"""Render one probabilistic 12-hour forecast as a GIF and a static summary figure.

Columns: observed surface current | forecast mean | absolute error | predicted
uncertainty (sigma of speed). Arrows show current direction.

Usage:
    python scripts/render_forecast.py --data data/cmems_iberia_uv_2024_2025.nc \
        --ckpt weights/unet_convlstm_curriculum.pt --issue-time 2025-10-20T06:00
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from currentcast.checkpoint import load_checkpoint
from currentcast.data import load_cmems_uv
from currentcast.forecast import rollout
from currentcast.plotting import ERROR_CMAP, INK, LAND, SECONDARY, SPEED_CMAP, apply_style


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", default="weights/unet_convlstm_curriculum.pt")
    p.add_argument("--issue-time", required=True, help="First forecast hour, e.g. 2025-10-20T06:00")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--out", default="docs/media")
    p.add_argument("--regrid", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats, seq_len, _ = load_checkpoint(args.ckpt, device)

    uv_raw, times, lat, lon = load_cmems_uv(args.data, regrid_hw=(args.regrid, args.regrid), fillna=None)
    land = np.isnan(uv_raw).any(axis=(0, 1))
    uv = np.nan_to_num(uv_raw, nan=0.0)
    t0 = int(np.searchsorted(times, np.datetime64(args.issue_time)))

    x = stats.normalize(torch.from_numpy(uv[t0 - seq_len:t0])[None]).to(device)
    with torch.inference_mode():
        mu_n, logvar_n = rollout(model, x, args.horizon)
    mu = stats.denormalize(mu_n[0]).cpu().numpy()                                  # (L, 2, H, W)
    sig = stats.denormalize_sigma(torch.exp(0.5 * logvar_n[0])).cpu().numpy()
    obs = uv[t0:t0 + args.horizon]

    speed = lambda f: np.hypot(f[:, 0], f[:, 1])  # noqa: E731
    fields = {
        "Observed (CMEMS)": speed(obs),
        "Forecast": speed(mu),
        "Absolute error": np.hypot(mu[:, 0] - obs[:, 0], mu[:, 1] - obs[:, 1]),
        "Predicted uncertainty (σ)": np.hypot(sig[:, 0], sig[:, 1]),
    }
    for f in fields.values():
        f[:, land] = np.nan
    vmax_speed = float(np.nanpercentile(fields["Observed (CMEMS)"], 99.5))
    vmax_err = float(np.nanpercentile(np.concatenate([fields["Absolute error"], fields["Predicted uncertainty (σ)"]]),
                                      99.5))
    norms = [(SPEED_CMAP, vmax_speed), (SPEED_CMAP, vmax_speed), (ERROR_CMAP, vmax_err), (ERROR_CMAP, vmax_err)]
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    q = 4  # arrow subsampling
    LON, LAT = np.meshgrid(lon, lat)
    vectors = [obs, mu, None, None]

    def draw(axes, k, cbar=False, titles=True):
        ims = []
        for ax, (title, f), (cmap, vmax), vec in zip(axes, fields.items(), norms, vectors, strict=True):
            ax.clear()
            ax.set_facecolor(LAND)
            im = ax.imshow(f[k], origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest", aspect="auto")
            if vec is not None:
                u, v = vec[k, 0].copy(), vec[k, 1].copy()
                u[land], v[land] = np.nan, np.nan
                ax.quiver(LON[::q, ::q], LAT[::q, ::q], u[::q, ::q], v[::q, ::q], color=INK, alpha=0.55,
                          scale=vmax_speed * 22, width=0.004, headwidth=3.5)
            ims.append(im)
            if titles:
                ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if cbar:
                add_colorbar(im, ax)
        return ims

    def add_colorbar(im, ax):
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
        cb.set_label("m/s", color=SECONDARY, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8)
        cb.locator = matplotlib.ticker.MaxNLocator(4)
        cb.update_ticks()

    stamp = str(times[t0])[:13].replace("T", " ") + ":00 UTC"

    # --- GIF -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), dpi=72)
    draw(axes, 0, cbar=True)
    sup = fig.suptitle("", x=0.01, ha="left", fontsize=13, fontweight="bold")

    def frame(k):
        draw(axes, k)
        sup.set_text(f"12-hour probabilistic forecast, issued {stamp}   ·   lead time +{k + 1} h")
        return []

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    frames = list(range(args.horizon)) + [args.horizon - 1] * 4   # hold the last frame
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, frame, frames=frames, blit=False).save(out / "forecast.gif", writer=PillowWriter(fps=2.5))
    plt.close(fig)

    # --- Static: rows at +1, +6, +12 h ----------------------------------------
    leads = [0, 5, args.horizon - 1]
    fig, grid = plt.subplots(len(leads), 4, figsize=(13, 3.4 * len(leads)), dpi=150)
    for i, (row, k) in enumerate(zip(grid, leads, strict=True)):
        ims = draw(row, k, titles=(i == 0))
        row[0].set_ylabel(f"+{k + 1} h", fontsize=13, fontweight="bold", color=INK, rotation=0, labelpad=28)
    fig.suptitle(f"Forecast issued {stamp}", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for j, im in enumerate(ims):   # one colorbar per column, spanning all rows
        add_colorbar(im, grid[:, j].tolist())
    fig.savefig(out / "forecast_panels.png")
    plt.close(fig)
    print(f"Wrote {out / 'forecast.gif'} and {out / 'forecast_panels.png'}")


if __name__ == "__main__":
    main()
