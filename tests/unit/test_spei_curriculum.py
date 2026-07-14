"""Unit tests for SPEI-proxy curriculum helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.spei_curriculum import (
    ACYD_PRECIP_MEAN_FEATURE_INDEX,
    cumulative_curriculum_stages,
    sort_by_random_order,
    sort_by_spei_difficulty,
    spei_proxy_difficulty,
    train_staged_curriculum_batched,
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


def test_spei_difficulty_rejects_bad_shape():
    with pytest.raises(ValueError, match="2-d"):
        spei_proxy_difficulty(np.zeros(3))
    with pytest.raises(ValueError, match="out of range"):
        spei_proxy_difficulty(np.zeros((2, 3)), precip_col=9)


def test_sort_by_random_order_is_seeded():
    x = np.arange(10, dtype=np.float64).reshape(10, 1)
    y = np.arange(10, dtype=np.float64)
    a, _ = sort_by_random_order(x, y, seed=7)
    b, _ = sort_by_random_order(x, y, seed=7)
    c, _ = sort_by_random_order(x, y, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cumulative_stages():
    x = np.arange(8, dtype=np.float64).reshape(8, 1)
    y = np.zeros(8, dtype=np.float64)
    stages = cumulative_curriculum_stages(x, y, n_stages=4)
    assert [len(s[0]) for s in stages] == [2, 4, 6, 8]
    with pytest.raises(ValueError, match="n_stages"):
        cumulative_curriculum_stages(x, y, n_stages=0)


def test_train_staged_curriculum_batched_smoke():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(x)).squeeze(-1)

    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(32, 2)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    x_val = rng.normal(size=(8, 2)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)
    metrics = train_staged_curriculum_batched(
        Tiny(),
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id="test_spei",
        model_name="tiny",
        n_stages=2,
        epochs_per_stage=1,
        refine_epochs=1,
        lr=0.05,
        batch_size=8,
        weight_decay=0.0,
        seed=1,
        profile="ci",
    )
    assert "val_roc_auc" in metrics
    assert isinstance(metrics["stage_metrics"], list)
    assert metrics["n_params"] >= 1
