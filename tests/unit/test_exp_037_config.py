"""Unit tests for exp_037 hybrid head configuration."""

from src.training.config import load_experiment_config


def test_exp_037_ci_profile():
    cfg = load_experiment_config("exp_037_hybrid_nano_higgs", profile="ci")
    assert cfg["exp_id"] == "exp_037"
    assert cfg["checkpoint_exp_id"] == "exp_032"
    assert cfg["n_qubits"] == 4
    assert cfg["n_train_rows"] == 8000
    assert cfg["min_vs_classical_pp"] == -2.0


def test_exp_037_publication_profile():
    cfg = load_experiment_config("exp_037_hybrid_nano_higgs", profile="publication")
    assert cfg["n_train_rows"] == 50000
    assert cfg["save_checkpoints"] is True
    assert cfg["min_vs_classical_pp"] == -1.0
