"""Unit tests for exp_075 GV-ALR hybrid NIHR configuration."""

from src.training.config import load_experiment_config


def test_exp_075_ci_profile():
    cfg = load_experiment_config("exp_075_adaptive_hybrid_nihr", profile="ci")
    assert cfg["exp_id"] == "exp_075"
    assert cfg["dataset_id"] == "nihr_cv_synthetic_v1"
    assert cfg["checkpoint_exp_id"] == "exp_069"
    assert cfg["fixed_epochs"] == 4
    assert cfg["adaptive_epochs"] == 2
    assert cfg["max_pr_delta_pp"] == 0.5


def test_exp_075_publication_profile():
    cfg = load_experiment_config("exp_075_adaptive_hybrid_nihr", profile="publication")
    assert cfg["fixed_epochs"] == 8
    assert cfg["adaptive_epochs"] == 5
    assert cfg["max_pr_delta_pp"] == 0.3
    assert cfg["n_train_rows"] == 50000
    assert cfg["n_val_rows"] == 15000
