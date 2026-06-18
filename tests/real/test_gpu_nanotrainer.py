"""Real-hardware tests — train on actual datasets with CUDA when available.

Run locally before release: ``make check-real``
Skipped in CI when no NVIDIA GPU is present.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Fail, Ok

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


def test_real_classical_mlp_breast_cancer(tmp_path, monkeypatch):
    """Classical MLP on full Wisconsin breast cancer — no mocks."""
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    dto = TrainNanomodelDTO(
        model_name="classical_mlp",
        dataset="breast_cancer",
        profile="ci",
        epochs=8,
        seed=42,
        exp_id="real_gate_classical",
    )
    outcome = train_execute(dto)
    assert isinstance(outcome, Ok), getattr(outcome.error, "message", outcome) if isinstance(outcome, Fail) else outcome
    result = outcome.value

    assert result.accuracy >= 0.80
    assert result.elapsed_s > 0
    assert result.n_params > 0

    log_lines = (tmp_path / "experiments.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert log_lines
    record = json.loads(log_lines[-1])
    assert record["exp_id"] == "real_gate_classical"
    assert record.get("test_accuracy") == pytest.approx(result.accuracy, rel=1e-5)


def test_real_hybrid_sandwich_breast_cancer(tmp_path, monkeypatch):
    """Hybrid quantum model on real breast cancer — PennyLane CPU + classical CUDA path."""
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        profile="ci",
        epochs=8,
        seed=42,
        exp_id="real_gate_hybrid",
    )
    outcome = train_execute(dto)
    assert isinstance(outcome, Ok), getattr(outcome.error, "message", outcome) if isinstance(outcome, Fail) else outcome
    result = outcome.value

    assert result.accuracy >= 0.80
    assert result.n_params > 0


def test_real_hybrid_sandwich_pima_diabetes(tmp_path, monkeypatch):
    """OpenML Pima dataset — real generalization path (exp_025 family)."""
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset="pima_diabetes",
        profile="ci",
        epochs=10,
        seed=7,
        exp_id="real_gate_pima",
    )
    outcome = train_execute(dto)
    assert isinstance(outcome, Ok), getattr(outcome.error, "message", outcome) if isinstance(outcome, Fail) else outcome
    result = outcome.value

    assert 0.60 <= result.accuracy <= 0.90
