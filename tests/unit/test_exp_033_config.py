"""Unit tests for exp_033 serve parity configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_033_ci_profile():
    cfg = load_experiment_config("exp_033_higgs_serve_parity", profile="ci")
    assert cfg["checkpoint_exp_id"] == "exp_032"
    assert cfg["model_name"] == "large_nano_mlp"
    assert cfg["dataset_id"] == "higgs_v1"
    assert cfg["n_rows"] == 500


def test_exp_033_publication_profile():
    cfg = load_experiment_config("exp_033_higgs_serve_parity", profile="publication")
    assert cfg["n_rows"] == 10000
