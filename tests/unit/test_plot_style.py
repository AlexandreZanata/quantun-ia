"""Unit tests for publication plot style."""

import matplotlib

matplotlib.use("Agg")

from src.training.plot_style import (
    FIGSIZE_SINGLE,
    PALETTE,
    apply_publication_style,
    new_figure,
)


def test_palette_has_eight_colors():
    assert len(PALETTE) == 8


def test_apply_publication_style_sets_rcparams():
    apply_publication_style()
    assert matplotlib.rcParams["savefig.dpi"] == 300
    assert matplotlib.rcParams["axes.spines.top"] is False


def test_new_figure_returns_axes():
    fig, ax = new_figure()
    assert fig.get_size_inches()[0] == FIGSIZE_SINGLE[0]
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
