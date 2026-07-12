"""Unit tests for exp_090 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_090_publication_config():
    cfg = load_experiment_config("exp_090_multicrop_joint_nano", profile="publication")
    assert cfg["include_crop_indicator"] is True
    assert float(cfg["min_vs_solo_pp"]) == -0.5
    assert cfg["n_train_rows_maize"] == 0
    assert cfg["epochs"] == 12


def test_exp_090_ci_config():
    cfg = load_experiment_config("exp_090_multicrop_joint_nano", profile="ci")
    assert cfg["n_train_rows_maize"] == 50
    assert cfg["n_train_rows_soy"] == 50
    assert cfg["epochs"] == 2
    assert float(cfg["min_vs_solo_pp"]) == -100.0
