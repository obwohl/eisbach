"""Shared plot style for the TimesFM experiments.

Categorical hues are taken in fixed slot order from a palette validated for
colour-vision deficiency; series are also direct-labelled, never colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_2 = "#5c5b55"
GRID = "#e4e3dd"

#: Fixed slot order. Never cycled, never reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITICAL = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 130,
})


def finish(fig, name: str, *, title: str, subtitle: str = "") -> Path:
    head = 0.86 if subtitle else 0.92
    fig.tight_layout(rect=(0, 0, 1, head))
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12, color=INK, weight="bold")
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9, color=INK_2, wrap=True)
    path = PLOTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
