"""Unit tests for exp_083 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_083_publication_config():
    cfg = load_experiment_config("exp_083_conventional_acyd_maize_baselines", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert float(cfg["min_advantage_pp"]) == 0.5


def test_exp_083_ci_config():
    cfg = load_experiment_config("exp_083_conventional_acyd_maize_baselines", profile="ci")
    assert float(cfg["min_advantage_pp"]) == 0.0
