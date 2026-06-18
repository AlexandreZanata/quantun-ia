"""Unit tests — exp_031 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config
from src.training.curriculum import curriculum_total_epochs


def test_exp_031_ci_config():
    cfg = load_experiment_config("exp_031_curriculum_clinical", profile="ci")
    assert cfg["dataset"] == "breast_cancer"
    assert len(cfg["seeds"]) == 3
    assert cfg["curriculum_stages"] == 2
    assert curriculum_total_epochs(cfg) == 12


def test_exp_031_publication_config():
    cfg = load_experiment_config("exp_031_curriculum_clinical", profile="publication")
    assert len(cfg["seeds"]) == 10
    assert cfg["min_advantage_pp"] == 0.0
    assert curriculum_total_epochs(cfg) == 60
