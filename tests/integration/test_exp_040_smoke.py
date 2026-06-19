"""Integration smoke test — exp_040 full-scale ablation (fast slice)."""

from __future__ import annotations

import pytest

import experiments.exp_040_full_scale_ablation_higgs.run as run_mod
from src.training.config import load_experiment_config

pytestmark = pytest.mark.real


def _fast_cfg() -> dict:
    cfg = load_experiment_config("exp_040_full_scale_ablation_higgs", profile="full_scale")
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


def test_exp_040_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(
        "experiments.exp_036_method_ablation_higgs.run.load_experiment_config",
        lambda key, profile=None: _fast_cfg(),
    )

    result = run_mod.run_exp_040(verbose=False, require_cuda=False)

    assert result.n_seeds == 1
    assert result.n_train_rows == 5000
    assert len(result.mean_auc_by_method) == 4
