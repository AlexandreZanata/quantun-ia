"""Integration smoke test — exp_011 UCI perceptron under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_011_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    return data.get("exp_011", {"perceptron": {"mean_min": 0.5, "mean_max": 1.0, "per_seed_min": 0.3, "per_seed_max": 1.0}})


def test_exp_011_ci_profile_holdout_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_011_ci(log_path=log_file)

    assert "perceptron" in results
    accs = results["perceptron"]
    assert len(accs) == 2
    mean_acc = sum(accs) / len(accs)

    bounds = golden_bounds["perceptron"]
    assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"]

    assert log_file.exists()
