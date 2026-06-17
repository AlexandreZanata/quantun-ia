"""Integration smoke test — exp_023 encoding×backend interaction under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_023_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    return json.loads(GOLDEN_PATH.read_text())["exp_023"]


def test_exp_023_ci_encoding_backend_smoke(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_023_ci(log_path=log_file)

    assert "angle_default" in results
    angle_default = results["angle_default"]
    assert len(angle_default) == 2
    angle_mean = sum(angle_default) / len(angle_default)
    angle_bounds = golden_bounds["angle_default"]
    assert angle_bounds["mean_min"] <= angle_mean <= angle_bounds["mean_max"]

    for key in ("amplitude_default", "angle_lightning", "amplitude_lightning"):
        if key not in results:
            continue
        accs = results[key]
        assert len(accs) == len(angle_default)
        mean = sum(accs) / len(accs)
        bounds = golden_bounds[key]
        assert bounds["mean_min"] <= mean <= bounds["mean_max"]

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines() if line.strip()]
    assert any(rec.get("exp_id") == "exp_023" for rec in lines)
