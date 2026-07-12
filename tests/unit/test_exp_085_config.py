"""Unit tests for exp_085 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_085_publication_config():
    cfg = load_experiment_config("exp_085_sample_efficiency_agro", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["epochs"] == 12
    assert cfg["fractions"] == [0.01, 0.05, 0.2, 1.0]
    assert int(cfg["min_budget_wins"]) == 2


def test_exp_085_ci_config():
    cfg = load_experiment_config("exp_085_sample_efficiency_agro", profile="ci")
    assert cfg["n_train_rows"] == 5000
    assert cfg["fractions"] == [0.2, 1.0]
    assert int(cfg["min_budget_wins"]) == 0
