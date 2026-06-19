"""Unit tests for exp_076 configuration."""

from src.training.config import load_experiment_config


def test_exp_076_publication_config():
    cfg = load_experiment_config("exp_076_conventional_nihr_baselines", profile="publication")
    assert cfg["exp_id"] == "exp_076"
    assert cfg["dataset_id"] == "nihr_cv_synthetic_v1"
    assert cfg["primary_metric"] == "pr_auc"
    assert float(cfg["min_advantage_pp"]) == 0.5


def test_exp_076_ci_config():
    cfg = load_experiment_config("exp_076_conventional_nihr_baselines", profile="ci")
    assert float(cfg["min_advantage_pp"]) <= 0.0
