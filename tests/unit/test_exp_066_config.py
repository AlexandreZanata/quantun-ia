"""Unit tests for exp_066 noise-reg ACYD configuration."""

from src.training.config import load_experiment_config


def test_exp_066_ci_profile():
    cfg = load_experiment_config("exp_066_noise_reg_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_066"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["n_train_rows"] == 5000
    assert cfg["n_val_rows"] == 1000
    assert cfg["epochs"] == 4
    assert cfg["min_auc_advantage_pp"] == -10.0
    assert cfg["depolarizing_p"] == 0.03


def test_exp_066_publication_profile():
    cfg = load_experiment_config("exp_066_noise_reg_acyd", profile="publication")
    assert cfg["n_train_rows"] == 0
    assert cfg["epochs"] == 10
    assert cfg["min_auc_advantage_pp"] == 0.5
    assert cfg["save_checkpoints"] is True
