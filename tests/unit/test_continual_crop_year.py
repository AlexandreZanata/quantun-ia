"""Unit tests for continual crop-year helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from src.data.continual_crop_year import rows_for_year
from src.training.continual_year import (
    mean_backward_auc,
    sklearn_auc,
    train_continual_by_year,
    train_joint,
)


def test_rows_for_year():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    years = np.array([2000, 2000, 2001, 2001, 2002, 2002], dtype=np.int32)
    x_y, y_y = rows_for_year(x, y, years, 2001)
    assert x_y.shape == (2, 2)
    assert list(y_y) == [0.0, 1.0]


def test_mean_backward_auc_runs():
    class Ranker(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(x[:, 0])

    model = Ranker()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    years = np.array([2000] * 20 + [2001] * 20, dtype=np.int32)
    auc = mean_backward_auc(model, x, y, years, [2000, 2001])
    assert 0.0 <= auc <= 1.0
    assert np.isnan(mean_backward_auc(model, x, y, years, []))


def test_sklearn_auc():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3)).astype(np.float64)
    y = (x[:, 0] > 0).astype(np.float64)
    model = LogisticRegression(max_iter=200).fit(x, y)
    assert 0.5 <= sklearn_auc(model, x, y) <= 1.0
    assert sklearn_auc(model, x, np.zeros_like(y)) == 0.5


def test_train_joint_and_continual_smoke():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(x)).squeeze(-1)

    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(40, 3)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    years = np.array([2000] * 20 + [2001] * 20, dtype=np.int32)
    x_val = rng.normal(size=(10, 3)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)

    joint = train_joint(
        Tiny(),
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id="test_joint",
        model_name="tiny",
        epochs=1,
        lr=0.05,
        batch_size=8,
        weight_decay=0.0,
        seed=2,
        profile="ci",
    )
    assert 0.0 <= joint <= 1.0

    val_auc, backward = train_continual_by_year(
        Tiny(),
        x_train,
        y_train,
        years,
        x_val,
        y_val,
        (2000, 2001),
        exp_id="test_cont",
        model_name="tiny",
        epochs_per_year=1,
        lr=0.05,
        batch_size=8,
        weight_decay=0.0,
        seed=2,
        profile="ci",
    )
    assert 0.0 <= val_auc <= 1.0
    assert np.isfinite(backward) or np.isnan(backward)
