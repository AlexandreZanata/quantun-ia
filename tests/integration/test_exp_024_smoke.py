"""Integration smoke test — exp_024 QuantumNano-BC under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_024_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["exp_024"]


def test_exp_024_ci_quantum_nano_bc_smoke(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    results = run_exp_024_ci(log_path=log_file)

    for model_name, bounds in golden_bounds.items():
        assert model_name in results, f"missing model {model_name}"
        mean_acc = sum(results[model_name]) / len(results[model_name])
        assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"]

    classical_keys = [k for k in results if k.startswith("classical_matched")]
    assert classical_keys, "expected parameter-matched classical baseline"
    for key in classical_keys:
        mean_acc = sum(results[key]) / len(results[key])
        assert 0.85 <= mean_acc <= 1.0

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines()]
    assert any(rec.get("exp_id") == "exp_024" for rec in lines)
