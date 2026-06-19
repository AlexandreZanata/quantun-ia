"""Unit tests for exp_069 configuration."""

from src.training.config import load_experiment_config


def test_exp_069_publication_config():
    cfg = load_experiment_config("exp_069_large_nano_nihr", profile="publication")
    assert cfg["exp_id"] == "exp_069"
    assert cfg["dataset_id"] == "nihr_cv_synthetic_v1"
    assert float(cfg["min_pr_advantage_pp"]) == 1.0


def test_exp_069_ci_config():
    cfg = load_experiment_config("exp_069_large_nano_nihr", profile="ci")
    assert float(cfg["min_pr_advantage_pp"]) <= 0.0
