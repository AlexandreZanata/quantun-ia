"""Unit tests for exp_092 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_092_publication_config():
    cfg = load_experiment_config("exp_092_histgb_distill_nano_maize", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["epochs"] == 12
    assert float(cfg["min_teacher_gap_pp"]) == 1.0
    assert float(cfg["distill_alpha"]) == 1.0


def test_exp_092_ci_config():
    cfg = load_experiment_config("exp_092_histgb_distill_nano_maize", profile="ci")
    assert cfg["n_train_rows"] == 5000
    assert cfg["epochs"] == 3
    assert float(cfg["min_teacher_gap_pp"]) == 100.0
