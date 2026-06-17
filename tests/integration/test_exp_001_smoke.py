"""Integration smoke test — exp_001 under CI profile with golden metric bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_001_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def test_exp_001_ci_profile_holdout_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_001_ci(log_path=log_file)

    assert "classical_8" in results
    accs = results["classical_8"]
    assert len(accs) == 2
    mean_acc = sum(accs) / len(accs)

    bounds = golden_bounds["exp_001"]["classical_8"]
    assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"], (
        f"mean holdout {mean_acc:.4f} outside golden range "
        f"[{bounds['mean_min']}, {bounds['mean_max']}]"
    )

    for acc in accs:
        assert bounds["per_seed_min"] <= acc <= bounds["per_seed_max"]

    assert log_file.exists()
    lines = [line for line in log_file.read_text().splitlines() if line.strip()]
    assert len(lines) >= 2
