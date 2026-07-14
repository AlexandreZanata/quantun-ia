"""Unit tests for exp_106 config."""

from src.training.config import load_experiment_config


def test_exp_106_ci_config():
    cfg = load_experiment_config("exp_106_latent_residual_qnn", profile="ci")
    assert cfg["exp_id"] == "exp_106"
    assert cfg["n_qubits"] == 4
    assert cfg["n_train"] == 32
    assert cfg["quantum_epochs"] >= 1
