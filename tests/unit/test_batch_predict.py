"""Unit tests for batch prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.application.batch_predict import (
    BatchPredictDTO,
    load_input_rows,
    max_probability_delta,
    run_batch_predict,
    write_output_csv,
)
from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Ok

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "breast_cancer_holdout.csv"


@pytest.fixture
def holdout_csv(tmp_path) -> Path:
    if FIXTURE.is_file():
        return FIXTURE
    pytest.skip("breast_cancer_holdout.csv fixture missing")


def test_load_input_rows_from_csv(holdout_csv):
    rows, columns = load_input_rows(holdout_csv)
    assert len(rows) == 569
    assert len(columns) == 30
    assert all(len(row) == 30 for row in rows)


def test_load_input_rows_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="features"):
        load_input_rows(path)


def test_max_probability_delta():
    assert max_probability_delta([0.5, 0.9], [0.5000001, 0.8999999]) < 1e-5


def test_run_batch_predict_matches_api_shape(tmp_path, monkeypatch, holdout_csv):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")

    train_execute(
        TrainNanomodelDTO(
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            profile="ci",
            epochs=6,
            seed=21,
            exp_id="batch_test",
            save_checkpoints=True,
        )
    )

    rows, _ = load_input_rows(holdout_csv)
    subset = rows[:12]
    dto = BatchPredictDTO(
        features=subset,
        exp_id="batch_test",
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        seed=21,
        chunk_size=4,
    )
    outcome = run_batch_predict(dto)
    assert isinstance(outcome, Ok)
    assert len(outcome.value.probabilities) == 12
    assert len(outcome.value.labels) == 12

    out_path = tmp_path / "out.csv"
    write_output_csv(out_path, outcome.value, source_input=str(holdout_csv))
    text = out_path.read_text(encoding="utf-8")
    assert "exp_id=batch_test" in text
    assert "probability" in text


def test_write_output_json(tmp_path):
    from src.application.batch_predict import BatchPredictResult, write_output_json

    result = BatchPredictResult(
        exp_id="exp",
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        seed=42,
        probabilities=[0.1, 0.9],
        labels=[0, 1],
        checkpoint_path="/tmp/ckpt",
        n_rows=2,
    )
    path = tmp_path / "out.json"
    write_output_json(path, result)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 2
    assert len(payload["probabilities"]) == 2
