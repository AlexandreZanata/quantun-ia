"""Integration smoke test — exp_018 feature fusion under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_018_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    return data.get(
        "exp_018",
        {"transformer_qnn_fusion": {"mean_min": 0.45, "mean_max": 1.0}},
    )


def test_exp_018_ci_fusion_holdout_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_018_ci(log_path=log_file)

    assert "transformer_qnn_fusion" in results
    accs = results["transformer_qnn_fusion"]
    assert len(accs) == 2
    mean_acc = sum(accs) / len(accs)
    bounds = golden_bounds["transformer_qnn_fusion"]
    assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"]
    assert log_file.exists()
