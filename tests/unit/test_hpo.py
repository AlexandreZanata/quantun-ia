"""Unit tests for Optuna HPO helpers."""

import pytest

from src.training.hpo import evaluate_uci_trial


@pytest.mark.slow
def test_evaluate_uci_trial_returns_float_in_range(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")

    score = evaluate_uci_trial(
        "exp_011_uci_tabular_qml",
        {"learning_rate": 0.02, "n_qubits": 4, "n_layers": 1, "hidden": 16},
        profile="ci",
        model_name="perceptron",
    )
    assert 0.0 <= score <= 1.0
