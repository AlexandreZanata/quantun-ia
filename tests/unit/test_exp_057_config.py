"""Unit tests for exp_057 configuration."""

from src.training.config import load_experiment_config


def test_exp_057_ci_profile():
    cfg = load_experiment_config("exp_057_param_shift_ablation", profile="ci")
    assert cfg["exp_id"] == "exp_057"
    assert cfg["n_layers"] == 3
    assert cfg["max_holdout_pp"] >= 1.0
    assert cfg["min_variance_ratio"] <= 2.0


def test_exp_057_publication_profile():
    cfg = load_experiment_config("exp_057_param_shift_ablation", profile="publication")
    assert cfg["max_holdout_pp"] == 1.0
    assert cfg["min_variance_ratio"] == 2.0
    assert len(cfg["seeds"]) == 10
