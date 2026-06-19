"""Unit tests for exp_074 entanglement schedule NIHR configuration."""

from src.training.config import load_experiment_config
from src.training.entangle_schedule import entanglement_for_stage


def test_exp_074_ci_profile():
    cfg = load_experiment_config("exp_074_entangle_schedule_nihr", profile="ci")
    assert cfg["exp_id"] == "exp_074"
    assert cfg["dataset_id"] == "nihr_cv_synthetic_v1"
    assert cfg["holdout_metric"] == "pr_auc"
    assert cfg["n_stages"] == 2
    assert len(cfg["seeds"]) == 1
    assert cfg["min_advantage_pp"] == -10.0
    assert entanglement_for_stage(0, cfg["n_stages"]) == "none"


def test_exp_074_publication_profile():
    cfg = load_experiment_config("exp_074_entangle_schedule_nihr", profile="publication")
    assert cfg["n_stages"] == 5
    assert len(cfg["seeds"]) == 3
    assert cfg["min_advantage_pp"] == 0.5
    assert cfg["n_train_rows"] == 10000
    assert cfg["n_val_rows"] == 3000
