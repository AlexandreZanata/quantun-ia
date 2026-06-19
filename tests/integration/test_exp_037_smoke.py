"""Integration smoke test — exp_037 hybrid head (fast slice, CPU)."""

from __future__ import annotations

import pytest

import experiments.exp_037_hybrid_nano_higgs.run as run_mod
from src.training.config import load_experiment_config

pytestmark = pytest.mark.real


def _fast_cfg(exp_key: str, profile: str | None = None) -> dict:
    cfg = load_experiment_config(exp_key, profile=profile)
    return {
        **cfg,
        "n_train_rows": 2000,
        "n_val_rows": 500,
        "epochs": 1,
        "batch_size": 128,
    }


def test_exp_037_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(
        run_mod,
        "load_experiment_config",
        lambda key, profile=None: _fast_cfg(key, profile),
    )

    result = run_mod.run_exp_037(profile="ci", verbose=False, require_cuda=False)

    assert result.n_train_rows == 2000
    assert result.n_val_rows == 500
    assert result.n_backbone_params > 1000
    assert 0 < result.n_trainable_params < result.n_backbone_params
    assert 0.5 < result.classical_val_auc <= 1.0
    assert 0.5 < result.hybrid_val_auc <= 1.0
