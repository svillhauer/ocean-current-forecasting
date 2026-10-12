"""Shared figure styling."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
LAND = "#d9d7cf"

COLORS = {
    "curriculum": "#2a78d6",   # categorical slot 1
    "onestep": "#eb6834",      # categorical slot 2
    "reference": "#898781",    # baseline: neutral, not a series hue
}

# Single-hue sequential ramps (light -> dark) for magnitude fields
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error", ["#fdf6f2", "#f9d3c1", "#f2a27f", "#eb6834", "#c4501f", "#8f3a15"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": SECONDARY,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def finish_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
