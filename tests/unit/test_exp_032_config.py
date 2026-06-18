"""Unit tests — exp_032 configuration."""

from src.training.config import load_experiment_config


def test_exp_032_ci_config():
    cfg = load_experiment_config("exp_032_large_nano_higgs", profile="ci")
    assert cfg["exp_id"] == "exp_032"
    assert cfg["dataset_id"] == "higgs_v1"
    assert cfg["n_train_rows"] == 50000
    assert cfg["min_params"] == 1_000_000


def test_exp_032_publication_config():
    cfg = load_experiment_config("exp_032_large_nano_higgs", profile="publication")
    assert cfg["epochs"] == 12
    assert cfg["min_auc_advantage_pp"] == 1.0
    assert cfg["batch_size"] == 2048
