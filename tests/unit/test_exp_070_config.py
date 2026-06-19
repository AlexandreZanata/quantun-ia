"""Unit tests for exp_070 configuration."""

from src.training.config import load_experiment_config


def test_exp_070_publication_config():
    cfg = load_experiment_config("exp_070_large_nano_gobug", profile="publication")
    assert cfg["exp_id"] == "exp_070"
    assert cfg["dataset_id"] == "code_defects_gobug_v1"
    assert float(cfg["min_pr_advantage_pp"]) == 2.0


def test_exp_070_ci_config():
    cfg = load_experiment_config("exp_070_large_nano_gobug", profile="ci")
    assert float(cfg["min_pr_advantage_pp"]) <= 0.0
