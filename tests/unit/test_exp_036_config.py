"""Unit tests for exp_036 methodology ablation configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_036_ci_profile():
    cfg = load_experiment_config("exp_036_method_ablation_higgs", profile="ci")
    assert cfg["dataset_id"] == "higgs_v1"
    assert cfg["n_train_rows"] == 50000
    assert len(cfg["seeds"]) == 3
    assert cfg["min_beat_baseline_pp"] == 0.5


def test_exp_036_publication_profile():
    cfg = load_experiment_config("exp_036_method_ablation_higgs", profile="publication")
    assert len(cfg["seeds"]) == 10
