"""Unit tests for exp_084b configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_084b_publication_config():
    cfg = load_experiment_config("exp_084b_residual_nano_soy_transfer", profile="publication")
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["exp_id"] == "exp_084b"
    assert cfg["epochs"] == 12
    assert float(cfg["min_auc_advantage_pp"]) == 0.5
    assert int(cfg["histgb_max_iter"]) == 100


def test_exp_084b_ci_config():
    cfg = load_experiment_config("exp_084b_residual_nano_soy_transfer", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert cfg["n_val_rows"] == 50
    assert cfg["epochs"] == 2
    assert float(cfg["min_auc_advantage_pp"]) == -100.0
