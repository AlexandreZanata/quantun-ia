from src.training.config import load_experiment_config


def test_exp_100_ci_config():
    cfg = load_experiment_config("exp_100_cycle2_grand_leaderboard", profile="ci")
    assert cfg["exp_id"] == "exp_100"
    assert cfg["registry_path"] == "config/cycle2_grand_leaderboard_registry.yaml"
    assert cfg["claim_win_delta_pp"] == 0.5


def test_exp_100_publication_paths():
    cfg = load_experiment_config("exp_100_cycle2_grand_leaderboard", profile="publication")
    assert cfg["json_out"] == "dist/leaderboards/cycle2_grand_leaderboard.json"
    assert cfg["latex_out"] == "paper/tables/cycle2_grand_leaderboard.tex"
