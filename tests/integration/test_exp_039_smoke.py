"""Integration smoke test — exp_039 regularized Synthea (fast slice)."""

from __future__ import annotations

import experiments.exp_039_synthea_regularized.run as run_mod
from src.training.config import load_experiment_config


def _fast_cfg(exp_key: str, profile: str | None = None) -> dict:
    cfg = load_experiment_config(exp_key, profile=profile)
    return {**cfg, "n_train_rows": 5000, "n_val_rows": 1000, "epochs": 2}


def test_exp_039_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr(
        run_mod,
        "load_experiment_config",
        lambda key, profile=None: _fast_cfg(key, profile),
    )

    result = run_mod.run_exp_039(profile="ci", verbose=False, require_cuda=False)

    assert result.n_train_rows == 5000
    assert result.dropout == 0.5
    assert result.n_params >= 1_000_000
    assert 0.0 <= result.nano_val_auc <= 1.0
