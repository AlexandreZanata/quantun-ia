"""Unit tests for exp_056 ladder configuration."""

from src.training.config import load_experiment_config


def test_exp_056_ci_profile():
    cfg = load_experiment_config("exp_056_reupload_curriculum_ladder", profile="ci")
    assert cfg["exp_id"] == "exp_056"
    assert len(cfg["rungs"]) == 3
    assert cfg["min_wins"] == 0
    assert cfg["min_rung_advantage_pp"] == -10.0


def test_exp_056_publication_profile():
    cfg = load_experiment_config("exp_056_reupload_curriculum_ladder", profile="publication")
    assert cfg["min_wins"] == 2
    assert len(cfg["rungs"]) == 3
    assert cfg["layer_ladder"] == [1, 2, 3]
