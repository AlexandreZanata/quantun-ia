"""Unit tests for exp_068b compound stress ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_068b_ci_profile():
    cfg = load_experiment_config("exp_068b_compound_stress_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_068b"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["label_mode"] == "compound_stress"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert float(cfg["min_vs_logistic_pp"]) == -5.0


def test_exp_068b_publication_profile():
    cfg = load_experiment_config("exp_068b_compound_stress_acyd", profile="publication")
    assert cfg["n_qubits"] == 4
    assert float(cfg["min_vs_logistic_pp"]) == 1.0
    assert cfg["save_checkpoints"] is True
