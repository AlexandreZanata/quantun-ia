"""Integration smoke test — exp_022 Nano Parity Bench under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_022_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    return json.loads(GOLDEN_PATH.read_text())["exp_022"]


def test_exp_022_hybrid_sandwich_beats_matched_classical(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_022_ci(log_path=log_file)

    assert "hybrid_sandwich" in results
    quantum_mean = float(results["quantum_mean"])
    classical_mean = float(results["classical_mean"])
    mean_diff = float(results["mean_diff"])
    param_delta = int(results["param_delta"])

    bounds = golden_bounds["hybrid_sandwich_breast_cancer"]
    assert bounds["quantum_mean_min"] <= quantum_mean <= bounds["quantum_mean_max"]
    assert bounds["classical_mean_min"] <= classical_mean <= bounds["classical_mean_max"]
    assert mean_diff > 0
    assert abs(param_delta) <= 10

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines() if line.strip()]
    assert any(rec.get("exp_id") == "exp_022" for rec in lines)
