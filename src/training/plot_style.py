"""Consistent matplotlib/seaborn style for publication figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-friendly palette (Okabe-Ito inspired)
PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

FIGSIZE_SINGLE = (6.0, 4.0)
FIGSIZE_WIDE = (8.0, 4.5)
DPI_SCREEN = 100
DPI_PUBLICATION = 300

RC_PARAMS: dict[str, object] = {
    "figure.dpi": DPI_SCREEN,
    "savefig.dpi": DPI_PUBLICATION,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def apply_publication_style() -> None:
    """Apply shared rcParams and seaborn theme for all publication plots."""
    mpl.rcParams.update(RC_PARAMS)
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
    sns.set_palette(PALETTE)
    # seaborn may reset spine visibility — re-apply publication defaults
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False


def new_figure(
    wide: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a styled figure with a single axes."""
    apply_publication_style()
    size = FIGSIZE_WIDE if wide else FIGSIZE_SINGLE
    return plt.subplots(figsize=size)
