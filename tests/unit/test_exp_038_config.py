"""Unit tests for exp_038 hybrid serve parity configuration."""

from src.training.config import load_experiment_config


def test_exp_038_ci_profile():
    cfg = load_experiment_config("exp_038_hybrid_serve_parity", profile="ci")
    assert cfg["exp_id"] == "exp_038"
    assert cfg["checkpoint_exp_id"] == "exp_037"
    assert cfg["model_name"] == "large_nano_hybrid"
    assert cfg["n_rows"] == 500


def test_exp_038_publication_profile():
    cfg = load_experiment_config("exp_038_hybrid_serve_parity", profile="publication")
    assert cfg["n_rows"] == 10000
