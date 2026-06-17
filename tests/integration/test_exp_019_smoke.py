"""Integration smoke test — exp_019 Nano Trainer path under CI profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute
from src.shared.result import Ok

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "regression" / "golden_ci.json"

SMOKE_MODELS = [
    ("perceptron", "breast_cancer"),
    ("transformer_qnn_fusion", "sequential_phase"),
]


@pytest.fixture
def golden_bounds() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    return data.get(
        "nano_train",
        {
            "perceptron": {"mean_min": 0.35, "mean_max": 1.0},
            "transformer_qnn_fusion": {"mean_min": 0.35, "mean_max": 1.0},
        },
    )


def test_exp_019_nanotrainer_ci_smoke(tmp_path, monkeypatch, golden_bounds):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    for model_name, dataset in SMOKE_MODELS:
        dto = TrainNanomodelDTO(
            model_name=model_name,
            dataset=dataset,
            profile="ci",
            exp_id="nano_train",
        )
        outcome = execute(dto)
        assert isinstance(outcome, Ok), f"{model_name}: {outcome}"
        acc = outcome.value.accuracy
        bounds = golden_bounds[model_name]
        assert bounds["mean_min"] <= acc <= bounds["mean_max"]

    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines()]
    assert len(lines) >= len(SMOKE_MODELS)
    assert all(rec.get("exp_id") == "nano_train" for rec in lines)
