"""Unit tests for sklearn baseline wrappers."""

import numpy as np
import torch

from src.classical.logistic_baseline import LogisticBaseline
from src.training import metrics as metrics_module


def test_logistic_baseline_fits_and_predicts(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_module, "LOGS_PATH", tmp_path / "experiments.jsonl")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    X_t = torch.tensor(X[:60])
    y_t = torch.tensor(y[:60])
    X_test = torch.tensor(X[60:])
    y_test = torch.tensor(y[60:])

    model = LogisticBaseline(input_dim=4)
    model.train(
        X_t,
        y_t,
        exp_id="test",
        model_name="logistic_seed42",
        seed=42,
        profile="ci",
        X_test=X_test,
        y_test=y_test,
    )

    metrics = model.evaluate(X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert model.count_sklearn_parameters() > 0


def test_sklearn_wrapper_mode_toggle():
    model = LogisticBaseline(input_dim=2)
    assert model.train(False) is model
    assert model.training is False
