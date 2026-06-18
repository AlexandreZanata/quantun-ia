"""Real gate — train hybrid, load checkpoint, predict holdout on RTX 4060."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.application.dto import PredictNanomodelDTO, TrainNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.application.train_nanomodel import execute as train_execute
from src.data.dataset_registry import get_dataset
from src.data.splits import split_train_test
from src.shared.result import Ok

pytestmark = pytest.mark.real


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for real gate — run on RTX 4060 workstation")
    if os.environ.get("QML_DEVICE", "auto") == "cpu":
        pytest.skip("QML_DEVICE=cpu — set QML_DEVICE=cuda for real gate")


@pytest.fixture(autouse=True)
def _cuda_env(monkeypatch):
    _require_cuda()
    monkeypatch.setenv("QML_DEVICE", "cuda")
    monkeypatch.setenv("MLFLOW_DISABLE", "1")


def test_real_checkpoint_train_and_predict_holdout(tmp_path, monkeypatch):
    """Train hybrid_sandwich on full breast cancer, predict 10 holdout rows."""
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    seed = 42
    exp_id = "real_app_shipped"
    train_outcome = train_execute(
        TrainNanomodelDTO(
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            profile="publication",
            epochs=20,
            seed=seed,
            exp_id=exp_id,
            save_checkpoints=True,
        )
    )
    assert isinstance(train_outcome, Ok), train_outcome
    train_result = train_outcome.value
    assert train_result.checkpoint_path
    assert train_result.accuracy >= 0.80

    X, y, _ = get_dataset("breast_cancer", random_state=seed)
    _, X_test, _, y_test = split_train_test(X, y, test_size=0.3, random_state=seed)
    rows = X_test[:10].tolist()

    pred_outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=exp_id,
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            seed=seed,
            features=rows,
        )
    )
    assert isinstance(pred_outcome, Ok), pred_outcome
    pred = pred_outcome.value
    holdout_acc = float(np.mean(np.array(pred.labels) == y_test[:10]))
    assert holdout_acc >= 0.7
    assert all(0.0 <= p <= 1.0 for p in pred.probabilities)
