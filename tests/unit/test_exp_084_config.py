"""Unit tests for exp_084 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_084_publication_config():
    cfg = load_experiment_config("exp_084_residual_ft_nano_maize", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["epochs"] == 12
    assert float(cfg["min_auc_advantage_pp"]) == 0.5
    assert int(cfg["histgb_max_iter"]) == 100


def test_exp_084_ci_config():
    cfg = load_experiment_config("exp_084_residual_ft_nano_maize", profile="ci")
    assert cfg["n_train_rows"] == 5000
    assert cfg["epochs"] == 3
    assert float(cfg["min_auc_advantage_pp"]) == -100.0
