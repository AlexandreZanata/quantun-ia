"""Unit tests for gradient-variance adaptive learning rate."""

import json

import torch
import torch.nn as nn

from src.classical.mlp import ClassicalNet
from src.training.adaptive_lr import (
    AdaptiveLRConfig,
    compute_lr_scale,
    step_gradient_variance,
    train_model_adaptive,
)


def test_compute_lr_scale_increases_when_variance_low():
    cfg = AdaptiveLRConfig(var_target=0.01, min_scale=0.5, max_scale=4.0)
    low_var_scale = compute_lr_scale(0.001, cfg)
    high_var_scale = compute_lr_scale(0.1, cfg)
    assert low_var_scale > high_var_scale
    assert 0.5 <= low_var_scale <= 4.0


def test_compute_lr_scale_clamps_extremes():
    cfg = AdaptiveLRConfig(var_target=0.01, min_scale=0.25, max_scale=2.0)
    assert compute_lr_scale(1e-15, cfg) == 2.0
    assert compute_lr_scale(100.0, cfg) == 0.25


def test_step_gradient_variance_non_negative():
    model = ClassicalNet(hidden=4)
    X = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,)).float()
    var = step_gradient_variance(model, X, y, nn.BCELoss())
    assert var >= 0.0


def test_train_model_adaptive_logs_lr_history(temp_log_file, sample_binary_data):
    X, y = sample_binary_data
    split = int(len(X) * 0.7)
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_test = torch.tensor(X[split:])
    y_test = torch.tensor(y[split:])

    model = ClassicalNet(hidden=8)
    cfg = AdaptiveLRConfig(base_lr=0.05, warmup_epochs=1, adapt_every=1)
    train_model_adaptive(
        model,
        X_train,
        y_train,
        "exp_test",
        "adaptive_model",
        epochs=4,
        config=cfg,
        X_test=X_test,
        y_test=y_test,
    )

    record = json.loads(temp_log_file.read_text().strip())
    assert record["adaptive_lr"] is True
    assert any("learning_rate" in epoch for epoch in record["history"])
