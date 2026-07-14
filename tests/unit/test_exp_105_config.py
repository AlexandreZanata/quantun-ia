"""Unit tests for exp_105 config."""

from src.training.config import load_experiment_config


def test_exp_105_ci_config():
    cfg = load_experiment_config("exp_105_image_difficulty_curriculum", profile="ci")
    assert cfg["exp_id"] == "exp_105"
    assert cfg["n_train"] == 32
    assert cfg["n_stages"] >= 2
    assert cfg["min_relative_fid_win"] == 0.05
