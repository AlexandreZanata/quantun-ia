"""Unit tests for exp_035 Synthea serve parity configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_035_ci_profile():
    cfg = load_experiment_config("exp_035_synthea_serve_parity", profile="ci")
    assert cfg["checkpoint_exp_id"] == "exp_034"
    assert cfg["dataset_id"] == "synthea_cv_risk_v1"
    assert cfg["n_rows"] == 500


def test_exp_035_publication_profile():
    cfg = load_experiment_config("exp_035_synthea_serve_parity", profile="publication")
    assert cfg["n_rows"] == 10000
