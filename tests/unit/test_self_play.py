"""Unit tests for self-play hard-example selection."""

import numpy as np
import torch

from src.classical.mlp import ClassicalNet
from src.training.self_play import select_hard_subset, self_play_train


def test_select_hard_subset_caps_by_frac():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)

    model = ClassicalNet(hidden=8)
    # Untrained model — many errors expected
    hard_X, hard_y = select_hard_subset(model, X, y, hard_frac=0.3, min_hard=5)

    assert hard_X is not None
    assert len(hard_X) <= 30
    assert len(hard_X) >= 5


def test_self_play_keeps_best_holdout(sample_binary_data):
    X, y = sample_binary_data
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = ClassicalNet(hidden=16)
    model.train(
        torch.tensor(X_train),
        torch.tensor(y_train),
        epochs=30,
        lr=0.05,
    )

    result = self_play_train(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        exp_id="exp_test",
        rounds=3,
        hard_frac=0.3,
        fine_tune_epochs=5,
        lr=0.01,
        revert_threshold=0.05,
    )

    assert result["best_holdout"] >= result["base_holdout"] - 0.05
    assert len(result["history"]) <= 3
