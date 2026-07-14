"""Unit tests for exp_101 config."""

from src.training.config import load_experiment_config


def test_exp_101_ci_config():
    cfg = load_experiment_config("exp_101_open_image_corpus_ingest", profile="ci")
    assert cfg["exp_id"] == "exp_101"
    assert cfg["n_smoke_rows"] == 8
