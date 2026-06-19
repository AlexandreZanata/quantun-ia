"""Integration smoke test — exp_036 methodology ablation (fast slice)."""

from __future__ import annotations

import pytest

import experiments.exp_036_method_ablation_higgs.run as run_mod
from src.training.config import load_experiment_config

pytestmark = pytest.mark.real


def _fast_cfg(exp_key: str, profile: str | None = None) -> dict:
    cfg = load_experiment_config(exp_key, profile=profile)
    return {
        **cfg,
        "seeds": [42],
        "n_train_rows": 5000,
        "n_val_rows": 1000,
        "baseline_epochs": 2,
        "curriculum_stages": 2,
        "epochs_per_stage": 1,
        "refine_epochs": 1,
        "adaptive_epochs": 2,
        "champion_epochs": 4,
    }


def test_exp_036_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(
        run_mod,
        "load_experiment_config",
        lambda key, profile=None: _fast_cfg(key, profile),
    )

    result = run_mod.run_exp_036(profile="ci", verbose=False, require_cuda=False)

    assert result.n_seeds == 1
    assert result.n_train_rows == 5000
    assert len(result.mean_auc_by_method) == 4
    assert all(0.0 < auc <= 1.0 for auc in result.mean_auc_by_method.values())
