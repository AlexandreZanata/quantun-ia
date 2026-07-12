"""Unit tests for exp_093 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_093_publication_config():
    cfg = load_experiment_config("exp_093_pqk_ridge_head", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert float(cfg["min_vs_logistic_pp"]) == 0.5
    assert cfg["n_qubits"] == 4
    assert cfg["n_train_rows"] == 15000


def test_exp_093_ci_config():
    cfg = load_experiment_config("exp_093_pqk_ridge_head", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert float(cfg["min_vs_logistic_pp"]) == -100.0
