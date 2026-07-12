"""Unit tests for exp_091 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_091_publication_config():
    cfg = load_experiment_config("exp_091_circuit_cut_6q", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_092"
    assert float(cfg["min_vs_classical_pp"]) == -1.0
    assert cfg["n_train_rows"] == 30000


def test_exp_091_ci_config():
    cfg = load_experiment_config("exp_091_circuit_cut_6q", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert float(cfg["min_vs_classical_pp"]) == -100.0
