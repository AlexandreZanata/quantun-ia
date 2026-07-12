"""Config tests for exp_079 quantum transfer HIGGS→ACYD."""

from src.training.config import load_experiment_config


def test_exp_079_ci_config():
    cfg = load_experiment_config("exp_079_quantum_transfer_higgs_to_acyd", profile="ci")
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert cfg["source_exp_id"] == "exp_037"
    assert cfg["n_train_rows"] == 5000
    assert float(cfg["max_transfer_advantage_pp"]) == 50.0


def test_exp_079_publication_config():
    cfg = load_experiment_config("exp_079_quantum_transfer_higgs_to_acyd", profile="publication")
    assert cfg["epochs"] == 8
    assert float(cfg["max_transfer_advantage_pp"]) == 0.5
    assert cfg["source_model_name"] == "large_nano_hybrid"
    assert cfg["save_checkpoints"] is True
