"""Integration tests for train_nanomodel use case."""

from __future__ import annotations

import json

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute
from src.shared.result import Fail, Ok


def test_execute_perceptron_breast_cancer_ci(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    dto = TrainNanomodelDTO(
        model_name="perceptron",
        dataset="breast_cancer",
        profile="ci",
        epochs=5,
        seed=42,
    )
    result = execute(dto)

    assert isinstance(result, Ok)
    r = result.value
    assert 0.5 <= r.accuracy <= 1.0
    assert r.exp_id == "nano_train"
    assert log_file.exists()
    lines = [json.loads(line) for line in log_file.read_text().strip().splitlines()]
    assert any(rec.get("exp_id") == "nano_train" for rec in lines)


def test_execute_invalid_pair_returns_fail():
    dto = TrainNanomodelDTO(
        model_name="perceptron",
        dataset="sequential_phase",
        profile="ci",
    )
    result = execute(dto)
    assert isinstance(result, Fail)
    assert result.error.code == "INVALID_PAIR"
