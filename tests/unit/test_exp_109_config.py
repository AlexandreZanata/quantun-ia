"""Config presence for exp_109."""

from src.training.config import load_experiment_config


def test_exp_109_config_ci_and_publication():
    cfg = load_experiment_config("exp_109_circuit_cut_latent_6q", profile="ci")
    assert cfg["exp_id"] == "exp_109"
    assert int(cfg["latent_dim"]) >= 4
    assert "parity_fid_slack" in cfg

    pub = load_experiment_config("exp_109_circuit_cut_latent_6q", profile="publication")
    assert float(pub["parity_fid_slack"]) == 1.0
