"""Integration smoke test — exp_021 PennyLane backend parity under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.ci_smoke import run_exp_021_ci

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    return data.get(
        "exp_021",
        {
            "quantum_default": {"mean_min": 0.2, "mean_max": 1.0, "per_seed_min": 0.15, "per_seed_max": 1.0},
            "quantum_lightning": {"mean_min": 0.2, "mean_max": 1.0, "per_seed_min": 0.15, "per_seed_max": 1.0},
        },
    )


def test_exp_021_ci_backend_smoke(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"

    results = run_exp_021_ci(log_path=log_file)

    assert "quantum_default" in results
    default_accs = results["quantum_default"]
    assert len(default_accs) == 2
    default_mean = sum(default_accs) / len(default_accs)
    default_bounds = golden_bounds["quantum_default"]
    assert default_bounds["mean_min"] <= default_mean <= default_bounds["mean_max"]

    if "quantum_lightning" in results:
        lightning_accs = results["quantum_lightning"]
        assert len(lightning_accs) == len(default_accs)
        lightning_mean = sum(lightning_accs) / len(lightning_accs)
        lightning_bounds = golden_bounds["quantum_lightning"]
        assert lightning_bounds["mean_min"] <= lightning_mean <= lightning_bounds["mean_max"]

    assert log_file.exists()
