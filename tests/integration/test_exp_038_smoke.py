"""Integration smoke test — exp_038 hybrid serve parity (fast slice)."""

from __future__ import annotations

import experiments.exp_038_hybrid_serve_parity.run as run_mod
from src.training.config import load_experiment_config


def _fast_cfg(exp_key: str, profile: str | None = None) -> dict:
    cfg = load_experiment_config(exp_key, profile=profile)
    return {**cfg, "n_rows": 50, "chunk_size": 32}


def test_exp_038_ci_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")

    monkeypatch.setattr(
        run_mod,
        "load_experiment_config",
        lambda key, profile=None: _fast_cfg(key, profile),
    )

    result = run_mod.run_exp_038(profile="ci", verbose=False, require_cuda=False)

    assert result.n_rows == 50
    assert result.max_delta_batch_api >= 0.0
    assert result.max_delta_tool_api >= 0.0
