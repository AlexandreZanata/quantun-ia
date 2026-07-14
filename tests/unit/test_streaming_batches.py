"""Unit tests for chronological streaming batches."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.streaming_batches import (
    chronological_batches,
    predict_proba,
    train_streaming_batches,
)


def test_chronological_batches_contiguous():
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)
    batches = chronological_batches(x, y, n_batches=5)
    assert len(batches) == 5
    assert [len(b[1]) for b in batches] == [2, 2, 2, 2, 2]
    # Order preserved
    assert list(batches[0][1]) == [0.0, 1.0]
    assert list(batches[-1][1]) == [8.0, 9.0]


def test_chronological_batches_remainder():
    x = np.zeros((11, 1), dtype=np.float64)
    y = np.arange(11, dtype=np.float64)
    batches = chronological_batches(x, y, n_batches=3)
    assert sum(len(b[1]) for b in batches) == 11
    assert len(batches[-1][1]) == 11 - 2 * (11 // 3)


def test_chronological_batches_rejects_bad_inputs():
    x = np.zeros((2, 1), dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    with pytest.raises(ValueError, match="n_batches"):
        chronological_batches(x, y, n_batches=0)
    with pytest.raises(ValueError, match="empty"):
        chronological_batches(x[:0], y[:0], n_batches=1)


def test_predict_proba_and_streaming_train():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.linear(x)).squeeze(-1)

    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(24, 2)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.float32)
    x_val = rng.normal(size=(8, 2)).astype(np.float32)
    y_val = (x_val[:, 0] > 0).astype(np.float32)
    model = Tiny()
    sizes = train_streaming_batches(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id="test_stream",
        model_name="tiny",
        n_batches=3,
        epochs_per_batch=1,
        lr=0.05,
        batch_size=8,
        weight_decay=0.0,
        seed=0,
        profile="ci",
    )
    assert len(sizes) == 3
    assert sum(sizes) == 24
    probs = predict_proba(model, x_val)
    assert probs.shape == (8,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
