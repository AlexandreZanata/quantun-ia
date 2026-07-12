"""Unit tests for exp_086 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_086_publication_config():
    cfg = load_experiment_config("exp_086_residual_qnn_head_maize", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_092"
    assert cfg["n_qubits"] == 4
    assert float(cfg["min_residual_vs_plain_pp"]) == 0.5
    assert cfg["n_train_rows"] == 50000


def test_exp_086_ci_config():
    cfg = load_experiment_config("exp_086_residual_qnn_head_maize", profile="ci")
    assert cfg["n_train_rows"] == 2000
    assert cfg["epochs"] == 2
    assert float(cfg["min_residual_vs_plain_pp"]) == -100.0
