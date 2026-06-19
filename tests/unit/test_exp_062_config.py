"""Unit tests for exp_062 hybrid ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_062_ci_profile():
    cfg = load_experiment_config("exp_062_hybrid_nano_acyd_soy", profile="ci")
    assert cfg["exp_id"] == "exp_062"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert cfg["n_qubits"] == 4
    assert float(cfg["min_vs_classical_pp"]) <= 0.0


def test_exp_062_publication_profile():
    cfg = load_experiment_config("exp_062_hybrid_nano_acyd_soy", profile="publication")
    assert float(cfg["min_vs_classical_pp"]) == -1.0
    assert cfg["save_checkpoints"] is True
