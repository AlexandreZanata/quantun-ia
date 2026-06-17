"""Integration smoke test — exp_017 poison topology under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_017_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    return data.get(
        "exp_017",
        {
            "hybrid_sandwich_poison_0": {
                "mean_min": 0.35,
                "mean_max": 1.0,
            },
            "hybrid_sandwich_poison_30": {
                "mean_min": 0.25,
                "mean_max": 1.0,
            },
        },
    )


def test_exp_017_ci_poison_holdout_in_golden_range(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_017_ci(log_path=log_file)

    for model_name in ("hybrid_sandwich_poison_0", "hybrid_sandwich_poison_30"):
        assert model_name in results
        accs = results[model_name]
        assert len(accs) == 2
        mean_acc = sum(accs) / len(accs)
        bounds = golden_bounds[model_name]
        assert bounds["mean_min"] <= mean_acc <= bounds["mean_max"]

    assert log_file.exists()
