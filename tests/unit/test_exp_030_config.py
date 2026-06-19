"""Unit tests — exp_030 configuration."""

from __future__ import annotations

from src.application.nanotrainer_config import load_nanotrainer_config, profile_settings
from src.training.config import load_experiment_config


def test_exp_030_ci_config():
    cfg = load_experiment_config("exp_030_publication_large", profile="ci")
    assert cfg["n_samples"] == 100
    assert len(cfg["seeds"]) == 3
    assert cfg["reference_seeds"] == 2
    assert cfg["parity_max_delta_pp"] == 5.0


def test_exp_030_publication_large_config():
    cfg = load_experiment_config("exp_030_publication_large", profile="publication_large")
    assert cfg["n_samples"] == 1000
    assert len(cfg["seeds"]) == 30
    assert cfg["reference_seeds"] == 10
    assert cfg["dataset"] == "circles"


def test_nanotrainer_publication_large_profile():
    cfg = load_nanotrainer_config()
    prof = profile_settings(cfg, "publication_large")
    assert prof["n_samples"] == 1000
    assert prof["epochs"] == 50
