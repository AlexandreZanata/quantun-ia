"""Unit tests for continual crop-year helpers."""

from __future__ import annotations

import numpy as np

from src.data.continual_crop_year import rows_for_year
from src.training.continual_year import mean_backward_auc


def test_rows_for_year():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    years = np.array([2000, 2000, 2001, 2001, 2002, 2002], dtype=np.int32)
    x_y, y_y = rows_for_year(x, y, years, 2001)
    assert x_y.shape == (2, 2)
    assert list(y_y) == [0.0, 1.0]


def test_mean_backward_auc_runs():
    import torch
    import torch.nn as nn

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
