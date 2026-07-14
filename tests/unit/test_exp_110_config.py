"""Config presence for exp_110."""

from src.training.config import load_experiment_config


def test_exp_110_config_ci_and_publication():
    cfg = load_experiment_config("exp_110_text_quantum_token_fusion", profile="ci")
    assert cfg["exp_id"] == "exp_110"
    assert int(cfg["n_qubits"]) == 4

    pub = load_experiment_config("exp_110_text_quantum_token_fusion", profile="publication")
    assert float(pub["min_clip_gap"]) == 0.5
