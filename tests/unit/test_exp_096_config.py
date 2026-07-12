"""Unit tests for exp_096 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_096_publication_config():
    cfg = load_experiment_config("exp_096_gobug_streaming_nano", profile="publication")
    assert cfg["dataset_id"] == "code_defects_gobug_v1"
    assert float(cfg["min_vs_joint_pp"]) == -1.0
    assert cfg["n_stream_batches"] == 8


def test_exp_096_ci_config():
    cfg = load_experiment_config("exp_096_gobug_streaming_nano", profile="ci")
    assert cfg["n_train_rows"] == 200
    assert float(cfg["min_vs_joint_pp"]) == -100.0
