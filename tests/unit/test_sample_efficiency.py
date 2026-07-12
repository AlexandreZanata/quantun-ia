"""Unit tests for sample-efficiency helpers."""

from __future__ import annotations

import numpy as np

from src.training.sample_efficiency import area_under_budget_curve, stratified_row_budget


def test_stratified_row_budget_full():
    x = np.random.randn(100, 4).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50, dtype=np.float32)
    xs, ys = stratified_row_budget(x, y, fraction=1.0)
    assert len(ys) == 100


def test_stratified_row_budget_fraction():
    x = np.random.randn(200, 3).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100, dtype=np.float32)
    xs, ys = stratified_row_budget(x, y, fraction=0.2, random_state=0)
    assert 30 <= len(ys) <= 50
    assert set(np.unique(ys).tolist()) == {0.0, 1.0}


def test_area_under_budget_curve():
    aulc = area_under_budget_curve([0.0, 1.0], [0.5, 0.7])
    assert abs(aulc - 0.6) < 1e-9
