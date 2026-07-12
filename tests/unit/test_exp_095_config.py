"""Unit tests for exp_095 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_095_publication_config():
    cfg = load_experiment_config("exp_095_cybench_maize_slice", profile="publication")
    assert cfg["dataset_id"] == "cybench_maize_us_v1"
    assert float(cfg["min_vs_histgb_pp"]) == -1.0
    assert cfg["epochs"] == 12


def test_exp_095_ci_config():
    cfg = load_experiment_config("exp_095_cybench_maize_slice", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert float(cfg["min_vs_histgb_pp"]) == -100.0
