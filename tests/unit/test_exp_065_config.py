"""Unit tests for exp_065 GV-ALR hybrid ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_065_ci_profile():
    cfg = load_experiment_config("exp_065_gv_alr_hybrid_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_065"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_060"
    assert cfg["n_train_rows"] == 5000
    assert cfg["n_val_rows"] == 1000
    assert cfg["fixed_epochs"] == 4
    assert cfg["adaptive_epochs"] == 2
    assert cfg["max_auc_delta_pp"] == 0.5
    assert cfg["max_epoch_fraction"] == 0.7


def test_exp_065_publication_profile():
    cfg = load_experiment_config("exp_065_gv_alr_hybrid_acyd", profile="publication")
    assert cfg["n_train_rows"] == 0
    assert cfg["fixed_epochs"] == 8
    assert cfg["adaptive_epochs"] == 5
    assert cfg["max_auc_delta_pp"] == 0.3
    assert cfg["save_checkpoints"] is True
