"""Unit tests for trainer holdout logging."""

import json

import torch

from src.classical.mlp import ClassicalNet
from src.training.trainer import train_model


def test_train_model_logs_holdout_accuracy(temp_log_file, sample_binary_data):
    X, y = sample_binary_data
    split = int(len(X) * 0.7)
    X_train = torch.tensor(X[:split])
    y_train = torch.tensor(y[:split])
    X_test = torch.tensor(X[split:])
    y_test = torch.tensor(y[split:])

    model = ClassicalNet(hidden=8)
    train_model(
        model,
        X_train,
        y_train,
        "exp_test",
        "holdout_model",
        epochs=3,
        lr=0.05,
        X_test=X_test,
        y_test=y_test,
    )

    record = json.loads(temp_log_file.read_text().strip())
    assert record["test_accuracy"] is not None
    assert record["eval_set"] == "holdout_test"
    assert 0.0 <= record["test_accuracy"] <= 1.0
