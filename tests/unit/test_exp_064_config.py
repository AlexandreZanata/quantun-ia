"""Unit tests for exp_064 entanglement schedule ACYD configuration."""

from src.training.config import load_experiment_config
from src.training.entangle_schedule import entanglement_for_stage


def test_exp_064_ci_profile():
    cfg = load_experiment_config("exp_064_entangle_schedule_acyd", profile="ci")
    assert cfg["exp_id"] == "exp_064"
    assert cfg["dataset_id"] == "acyd_soy_brazil_v1"
    assert cfg["holdout_metric"] == "roc_auc"
    assert cfg["n_stages"] == 2
    assert len(cfg["seeds"]) == 1
    assert cfg["min_advantage_pp"] == -10.0
    assert entanglement_for_stage(0, cfg["n_stages"]) == "none"


def test_exp_064_publication_profile():
    cfg = load_experiment_config("exp_064_entangle_schedule_acyd", profile="publication")
    assert cfg["n_stages"] == 5
    assert len(cfg["seeds"]) == 3
    assert cfg["min_advantage_pp"] == 0.5
    assert cfg["n_train_rows"] == 10000
    assert cfg["n_val_rows"] == 3000
    assert cfg["holdout_metric"] == "roc_auc"
