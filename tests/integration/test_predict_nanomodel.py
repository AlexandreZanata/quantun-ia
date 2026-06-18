"""Integration tests — load checkpoint and predict on holdout rows."""

from __future__ import annotations

import numpy as np
import pytest

from src.application.dto import PredictNanomodelDTO, TrainNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.application.train_nanomodel import execute as train_execute
from src.data.dataset_registry import get_dataset
from src.data.splits import split_train_test
from src.shared.result import Fail, Ok
from src.training.checkpoints import checkpoint_path, load_scaler


@pytest.fixture
def trained_hybrid_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        profile="ci",
        epochs=8,
        seed=42,
        exp_id="predict_test",
        save_checkpoints=True,
    )
    outcome = train_execute(dto)
    assert isinstance(outcome, Ok), outcome
    ckpt_dir = checkpoint_path("predict_test", "hybrid_sandwich_breast_cancer", 42)
    assert ckpt_dir.is_dir()
    assert (ckpt_dir / "best.pt").is_file()
    assert load_scaler(ckpt_dir) is not None
    return dto, outcome.value, ckpt_dir


def test_predict_holdout_rows_matches_eval(trained_hybrid_checkpoint):
    dto, train_result, _ = trained_hybrid_checkpoint
    X, y, _ = get_dataset(dto.dataset, random_state=dto.seed)
    _, X_test, _, y_test = split_train_test(
        X, y, test_size=dto.test_size, random_state=dto.seed
    )
    rows = X_test[:10].tolist()

    pred_outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            features=rows,
        )
    )
    assert isinstance(pred_outcome, Ok), pred_outcome
    result = pred_outcome.value

    assert len(result.probabilities) == 10
    assert len(result.labels) == 10
    assert all(0.0 <= p <= 1.0 for p in result.probabilities)
    assert all(label in (0, 1) for label in result.labels)

    holdout_acc = float(np.mean(np.array(result.labels) == y_test[:10]))
    assert holdout_acc >= 0.7


def test_predict_missing_checkpoint_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id="missing",
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            seed=99,
            features=[[0.1] * 30],
        )
    )
    assert isinstance(outcome, Fail)
    assert outcome.error.code == "CHECKPOINT_NOT_FOUND"
