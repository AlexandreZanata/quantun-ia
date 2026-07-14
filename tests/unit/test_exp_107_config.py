"""Unit tests for exp_107 config."""

from src.training.config import load_experiment_config


def test_exp_107_ci_config():
    cfg = load_experiment_config("exp_107_patch_amplitude_bottleneck", profile="ci")
    assert cfg["exp_id"] == "exp_107"
    assert cfg["n_qubits"] == 4
    assert cfg["bottleneck"] == 16
    assert 2 ** cfg["n_qubits"] == cfg["bottleneck"]
