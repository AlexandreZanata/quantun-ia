"""Unit tests for exp_063 ACYD warm-start configuration."""

from src.training.config import load_experiment_config
from src.training.quantum_warmstart import split_warmstart_epochs


def test_exp_063_ci_profile():
    cfg = load_experiment_config("exp_063_quantum_warmstart_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_063"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["n_qubits"] == 4
    assert cfg["n_train_rows"] == 5000
    assert cfg["n_val_rows"] == 1000
    assert len(cfg["seeds"]) == 1
    assert cfg["min_vs_e2e_pp"] == -10.0
    classical, quantum = split_warmstart_epochs(int(cfg["epochs"]), float(cfg["classical_fraction"]))
    assert classical + quantum == int(cfg["epochs"])


def test_exp_063_publication_profile():
    cfg = load_experiment_config("exp_063_quantum_warmstart_acyd", profile="publication")
    assert cfg["n_train_rows"] == 0
    assert len(cfg["seeds"]) == 3
    assert cfg["min_vs_e2e_pp"] == 0.5
    assert cfg["save_checkpoints"] is True
