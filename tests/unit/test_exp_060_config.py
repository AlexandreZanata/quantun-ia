"""Unit tests for exp_060 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_060_publication_config():
    cfg = load_experiment_config("exp_060_large_nano_acyd_soy", profile="publication")
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["epochs"] == 12
    assert float(cfg["min_auc_advantage_pp"]) == 2.0
    assert int(cfg["min_params"]) == 1_000_000


def test_exp_060_ci_config():
    cfg = load_experiment_config("exp_060_large_nano_acyd_soy", profile="ci")
    assert cfg["n_train_rows"] == 5000
    assert float(cfg["min_auc_advantage_pp"]) == 0.0
