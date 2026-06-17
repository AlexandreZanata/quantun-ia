"""Integration smoke test — exp_022 Nano Parity Bench under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.application.dto import NanoParityBenchDTO
from src.application.nano_parity_bench import execute
from src.shared.result import Ok

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"


@pytest.fixture
def golden_bounds() -> dict:
    return json.loads(GOLDEN_PATH.read_text())["exp_022"]


def test_exp_022_hybrid_sandwich_beats_matched_classical(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    dto = NanoParityBenchDTO(
        quantum_model="hybrid_sandwich",
        dataset="wine_binary",
        profile="ci",
        exp_id="exp_022",
        seeds=[42, 123],
        epochs=12,
    )
    outcome = execute(dto)
    assert isinstance(outcome, Ok), outcome
    result = outcome.value

    bounds = golden_bounds["hybrid_sandwich_wine_binary"]
    assert bounds["quantum_mean_min"] <= result.quantum_mean <= bounds["quantum_mean_max"]
    assert result.quantum_mean >= result.classical_mean
    assert result.comparison["mean_diff"] > 0
    assert abs(result.quantum_n_params - result.classical_n_params) <= 10

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines() if line.strip()]
    assert any(rec.get("exp_id") == "exp_022" for rec in lines)
