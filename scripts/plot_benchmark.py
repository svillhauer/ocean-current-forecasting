#!/usr/bin/env python3
"""Render README figures from results/benchmark.json (written by evaluate.py).

Usage:
    python scripts/plot_benchmark.py --results results/benchmark.json --out docs/media
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from currentcast.plotting import COLORS, INK, MUTED, SECONDARY, apply_style, finish_axes

SERIES = [
    # key, label, color, linestyle
    ("persistence", "Persistence", COLORS["reference"], (0, (4, 3))),
    ("onestep", "One-step training", COLORS["onestep"], "-"),
    ("curriculum", "Curriculum training", COLORS["curriculum"], "-"),
]


def lead_axis(ax, n):
    ax.set_xlim(0.6, n + 0.4)
    ax.set_xticks(range(1, n + 1))
    ax.set_xlabel("Forecast lead time (hours)")


def plot_rmse(res, out):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    n = len(res["persistence"]["rmse_vector"])
    leads = range(1, n + 1)
    for key, label, color, ls in SERIES:
        ax.plot(leads, res[key]["rmse_vector"], color=color, linestyle=ls, linewidth=2, marker="o",
                markersize=4.5, markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
    # End-of-line labels, nudged apart so close finishes don't collide
    ends = sorted(((res[k]["rmse_vector"][-1], lbl) for k, lbl, _, _ in SERIES), reverse=True)
    min_gap = 0.022
    placed = []
    for y, lbl in ends:
        y_text = min(y, placed[-1] - min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(f"{lbl}  {y:.3f}", (n, y), xytext=(n + 0.35, y_text), fontsize=9.5, color=INK, va="center")
    lead_axis(ax, n)
    ax.set_xlim(0.6, n + 3.6)
    ax.set_ylim(0, None)
    ax.set_ylabel("Vector RMSE (m/s)")
    ax.set_title("Forecast error vs. lead time", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    finish_axes(ax)
    fig.text(0.99, 0.01, "Held-out test period, Aug 2025 to Jan 2026 (3,498 forecasts), ocean pixels only",
             ha="right", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "rmse_vs_lead.png")
    plt.close(fig)


def plot_calibration(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    n = len(res["curriculum"]["coverage_1sigma"])
    leads = range(1, n + 1)
    for ax, k, nominal in ((axes[0], 1, 0.683), (axes[1], 2, 0.954)):
        ax.axhline(nominal * 100, color=SECONDARY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(n + 0.3, nominal * 100 + 1.5, f"ideal {nominal:.0%}", ha="right", va="bottom",
                fontsize=9, color=SECONDARY)
        for key, label, color, _ in SERIES[1:]:
            ax.plot(leads, [c * 100 for c in res[key][f"coverage_{k}sigma"]], color=color, linewidth=2,
                    marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
        lead_axis(ax, n)
        ax.set_title(f"Truth inside ±{k}σ", loc="left", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        finish_axes(ax)
    axes[0].set_ylabel("Coverage (%)")
    axes[1].legend(frameon=False, loc="center right", fontsize=10)
    fig.suptitle("Is the predicted uncertainty honest?", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "calibration.png")
    plt.close(fig)


def plot_spread(res, out):
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    n = len(res["curriculum"]["rmse_component"])
    leads = range(1, n + 1)
    for key, label, color, _ in SERIES[1:]:
        ax.plot(leads, res[key]["rmse_component"], color=color, linewidth=2, label=f"{label}: actual error")
        ax.plot(leads, res[key]["spread_component"], color=color, linewidth=2, linestyle=(0, (4, 3)),
                label=f"{label}: predicted σ")
    lead_axis(ax, n)
    ax.set_ylim(0, None)
    ax.set_ylabel("m/s (per component)")
    ax.set_title("Predicted uncertainty vs. actual error", loc="left", fontsize=14, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    finish_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "spread_vs_error.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/benchmark.json")
    p.add_argument("--out", default="docs/media")
    args = p.parse_args()
    apply_style()
    res = json.loads(Path(args.results).read_text())["models"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_rmse(res, out)
    plot_calibration(res, out)
    plot_spread(res, out)
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
