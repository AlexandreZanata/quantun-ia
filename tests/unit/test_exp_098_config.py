"""Unit tests for exp_098 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_098_publication_config():
    cfg = load_experiment_config("exp_098_continual_crop_year", profile="publication")
    assert float(cfg["min_vs_joint_pp"]) == -1.0
    assert cfg["epochs_per_year"] == 2


def test_exp_098_ci_config():
    cfg = load_experiment_config("exp_098_continual_crop_year", profile="ci")
    assert cfg["n_train_rows"] == 200
    assert float(cfg["min_vs_joint_pp"]) == -100.0
    assert cfg["max_feature_chunks"] == 3
