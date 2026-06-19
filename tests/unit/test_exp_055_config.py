"""Unit tests for exp_055 noise regularization configuration."""

from src.training.config import load_experiment_config


def test_exp_055_ci_profile():
    cfg = load_experiment_config("exp_055_noise_reg_gobug", profile="ci")
    assert cfg["exp_id"] == "exp_055"
    assert cfg["dataset_id"] == "code_defects_gobug_v1"
    assert cfg["n_train_rows"] == 5000
    assert cfg["depolarizing_p"] == 0.03
    assert cfg["min_pr_advantage_pp"] == -10.0


def test_exp_055_publication_profile():
    cfg = load_experiment_config("exp_055_noise_reg_gobug", profile="publication")
    assert cfg["n_train_rows"] == 0
    assert cfg["min_pr_advantage_pp"] == 0.5
    assert cfg["save_checkpoints"] is True
