"""Unit tests for exp_099 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_099_publication_config():
    cfg = load_experiment_config("exp_099_ssl_climate_pretrain", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert float(cfg["min_vs_scratch_pp"]) == 0.5
    assert cfg["pretrain_epochs"] == 8
    assert cfg["finetune_epochs"] == 12


def test_exp_099_ci_config():
    cfg = load_experiment_config("exp_099_ssl_climate_pretrain", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert float(cfg["min_vs_scratch_pp"]) == -100.0
