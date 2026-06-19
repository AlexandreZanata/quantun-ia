"""Unit tests for exp_071 hybrid GoBug configuration."""

from src.training.config import load_experiment_config


def test_exp_071_ci_profile():
    cfg = load_experiment_config("exp_071_hybrid_nano_gobug", profile="ci")
    assert cfg["exp_id"] == "exp_071"
    assert cfg["dataset_id"] == "code_defects_gobug_v1"
    assert cfg["checkpoint_exp_id"] == "exp_070"
    assert cfg["n_qubits"] == 4
    assert float(cfg["min_vs_classical_pp"]) <= 0.0


def test_exp_071_publication_profile():
    cfg = load_experiment_config("exp_071_hybrid_nano_gobug", profile="publication")
    assert float(cfg["min_vs_classical_pp"]) == -1.0
    assert cfg["save_checkpoints"] is True
