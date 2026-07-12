"""Unit tests for exp_094 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_094_publication_config():
    cfg = load_experiment_config("exp_094_hard_temporal_drift", profile="publication")
    assert cfg["train_max_year"] == 2016
    assert list(cfg["val_years"]) == [2017, 2018]
    assert cfg["test_min_year"] == 2022
    assert float(cfg["min_vs_histgb_pp"]) == -1.0
    assert cfg["epochs"] == 12


def test_exp_094_ci_config():
    cfg = load_experiment_config("exp_094_hard_temporal_drift", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert cfg["max_feature_chunks"] == 2
    assert float(cfg["min_vs_histgb_pp"]) == -100.0
