"""Unit tests for exp_068 grand comparison configuration."""

from src.training.config import load_experiment_config


def test_exp_068_ci_profile():
    cfg = load_experiment_config("exp_068_nano_grand_comparison", profile="ci")
    assert cfg["exp_id"] == "exp_068"
    assert cfg["registry_path"] == "config/grand_comparison_registry.yaml"
    assert float(cfg["claim_win_delta_pp"]) == 0.5


def test_exp_068_publication_profile():
    cfg = load_experiment_config("exp_068_nano_grand_comparison", profile="publication")
    assert cfg["json_out"] == "dist/leaderboards/nano_grand_comparison.json"
    assert cfg["latex_out"] == "paper/tables/grand_comparison.tex"
