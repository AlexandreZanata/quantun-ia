"""Unit tests for exp_088 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_088_publication_config():
    cfg = load_experiment_config("exp_088_shadow_features_nano_maize", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["n_qubits"] == 4
    assert cfg["n_shadow_features"] == 64
    assert float(cfg["min_vs_classical_pp"]) == -0.5
    assert float(cfg["min_vs_logistic_pp"]) == 2.0
    assert cfg["n_train_rows"] == 20000


def test_exp_088_ci_config():
    cfg = load_experiment_config("exp_088_shadow_features_nano_maize", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert cfg["n_val_rows"] == 50
    assert cfg["epochs"] == 2
    assert float(cfg["min_vs_classical_pp"]) == -100.0
