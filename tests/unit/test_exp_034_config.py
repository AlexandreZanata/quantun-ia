"""Unit tests for exp_034 Synthea CV training configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_034_ci_profile():
    cfg = load_experiment_config("exp_034_large_nano_synthea", profile="ci")
    assert cfg["dataset_id"] == "synthea_cv_risk_v1"
    assert cfg["n_train_rows"] == 50000
    assert cfg["min_auc_advantage_pp"] == 0.0


def test_exp_034_publication_profile():
    cfg = load_experiment_config("exp_034_large_nano_synthea", profile="publication")
    assert cfg["save_checkpoints"] is True
    assert cfg["min_auc_advantage_pp"] == -2.0
