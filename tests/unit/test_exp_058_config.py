"""Config smoke tests for exp_058."""

from src.training.config import load_experiment_config


def test_exp_058_config_ci():
    cfg = load_experiment_config("exp_058_conventional_higgs_baselines", profile="ci")
    assert cfg["exp_id"] == "exp_058"
    assert float(cfg["min_advantage_pp"]) == 0.0


def test_exp_058_config_publication():
    cfg = load_experiment_config("exp_058_conventional_higgs_baselines", profile="publication")
    assert float(cfg["min_advantage_pp"]) == 0.5
    assert int(cfg["histgb_max_iter"]) == 100
