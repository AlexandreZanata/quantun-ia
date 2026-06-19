"""Unit tests for exp_068a angle encoding ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_068a_ci_profile():
    cfg = load_experiment_config("exp_068a_angle_encoding_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_068a"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert float(cfg["min_vs_classical_pp"]) == 0.0
    assert float(cfg["min_vs_amplitude_pp"]) == 0.0


def test_exp_068a_publication_profile():
    cfg = load_experiment_config("exp_068a_angle_encoding_acyd", profile="publication")
    assert cfg["n_qubits"] == 4
    assert cfg["save_checkpoints"] is True
