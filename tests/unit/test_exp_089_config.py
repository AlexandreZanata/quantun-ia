"""Unit tests for exp_089 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_089_publication_config():
    cfg = load_experiment_config("exp_089_measurement_dropout_cal", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_092"
    assert float(cfg["measurement_dropout"]) == 0.2
    assert float(cfg["min_ece_relative_improvement"]) == 0.20
    assert cfg["n_train_rows"] == 30000


def test_exp_089_ci_config():
    cfg = load_experiment_config("exp_089_measurement_dropout_cal", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert float(cfg["min_ece_relative_improvement"]) == -1.0
