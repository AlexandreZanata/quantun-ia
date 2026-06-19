"""Unit tests for exp_077 configuration."""

from src.training.config import load_experiment_config


def test_exp_077_publication_config():
    cfg = load_experiment_config("exp_077_conventional_gobug_baselines", profile="publication")
    assert cfg["exp_id"] == "exp_077"
    assert cfg["dataset_id"] == "code_defects_gobug_v1"
    assert cfg["primary_metric"] == "pr_auc"
    assert float(cfg["min_advantage_pp"]) == 0.5


def test_exp_077_ci_config():
    cfg = load_experiment_config("exp_077_conventional_gobug_baselines", profile="ci")
    assert float(cfg["min_advantage_pp"]) <= 0.0
