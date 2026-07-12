"""Unit tests for SPEI-proxy curriculum helpers."""

from __future__ import annotations

import numpy as np

from src.training.spei_curriculum import (
    ACYD_PRECIP_MEAN_FEATURE_INDEX,
    cumulative_curriculum_stages,
    sort_by_spei_difficulty,
    spei_proxy_difficulty,
)


def test_precip_feature_index():
    assert ACYD_PRECIP_MEAN_FEATURE_INDEX == 9


def test_spei_sort_puts_wet_first():
    x = np.zeros((4, 12), dtype=np.float64)
    x[:, 9] = [0.0, 2.0, -1.0, 1.0]  # precip mean
    y = np.array([0, 1, 0, 1], dtype=np.float64)
    xs, ys = sort_by_spei_difficulty(x, y)
    # wet (2.0) first, dry (-1.0) last
    assert list(xs[:, 9]) == [2.0, 1.0, 0.0, -1.0]
    assert len(ys) == 4


def test_spei_difficulty_scores():
    x = np.zeros((3, 12), dtype=np.float64)
    x[:, 9] = [1.0, -2.0, 0.0]
    d = spei_proxy_difficulty(x)
    assert d[1] > d[2] > d[0]


def test_cumulative_stages():
    x = np.arange(8, dtype=np.float64).reshape(8, 1)
    y = np.zeros(8, dtype=np.float64)
    stages = cumulative_curriculum_stages(x, y, n_stages=4)
    assert [len(s[0]) for s in stages] == [2, 4, 6, 8]
