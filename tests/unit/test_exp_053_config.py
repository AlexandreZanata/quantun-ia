"""Unit tests for exp_053 entanglement schedule configuration."""

from src.training.config import load_experiment_config
from src.training.entangle_schedule import entanglement_for_stage


def test_exp_053_ci_profile():
    cfg = load_experiment_config("exp_053_entangle_schedule_bc", profile="ci")
    assert cfg["exp_id"] == "exp_053"
    assert cfg["dataset"] == "breast_cancer"
    assert cfg["reupload"] is True
    assert cfg["n_stages"] == 2
    assert len(cfg["seeds"]) == 1
    assert cfg["min_advantage_pp"] == -30.0
    assert entanglement_for_stage(0, cfg["n_stages"]) == "none"


def test_exp_053_publication_profile():
    cfg = load_experiment_config("exp_053_entangle_schedule_bc", profile="publication")
    assert cfg["n_stages"] == 5
    assert len(cfg["seeds"]) == 3
    assert cfg["min_advantage_pp"] == 1.0
