"""Unit tests for exp_087 configuration."""

from __future__ import annotations

from src.training.config import load_experiment_config


def test_exp_087_publication_config():
    cfg = load_experiment_config("exp_087_fourier_reupload_climate", profile="publication")
    assert cfg["dataset_id"] == "acyd_maize_brazil_v1"
    assert cfg["checkpoint_exp_id"] == "exp_092"
    assert cfg["n_qubits"] == 4
    assert list(cfg["reupload_layers"]) == [1, 2, 3]
    assert int(cfg["min_rung_wins"]) == 2
    assert cfg["n_train_rows"] == 30000


def test_exp_087_ci_config():
    cfg = load_experiment_config("exp_087_fourier_reupload_climate", profile="ci")
    assert cfg["n_train_rows"] == 50
    assert cfg["epochs"] == 1
    assert int(cfg["min_rung_wins"]) == 0
