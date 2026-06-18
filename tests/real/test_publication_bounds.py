"""Real gate — holdout accuracy within publication parity bands on RTX 4060."""

from __future__ import annotations

import os

import pytest
import torch

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Ok

pytestmark = pytest.mark.real

# Publication reference (30-seed profile, RTX 4060 — exp_024/025 results.md 2026-06-18)
EXP_024_HYBRID_CI = (0.970, 0.977)  # mean 97.4%
EXP_025_HYBRID_CI = (0.753, 0.771)  # mean 76.2%
VALIDATION_SEEDS = [42, 123, 456, 789, 1024, 1337, 2048, 3001, 4096, 5000]


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


def _train_hybrid(
    *,
    dataset: str,
    seed: int,
    exp_id: str,
    tmp_path,
    monkeypatch,
) -> float:
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset=dataset,
        profile="publication",
        epochs=50,
        seed=seed,
        exp_id=exp_id,
        save_checkpoints=True,
    )
    outcome = train_execute(dto)
    assert isinstance(outcome, Ok), outcome
    return outcome.value.accuracy


def test_exp_024_hybrid_within_publication_band(tmp_path, monkeypatch):
    """Ten fresh seeds: hybrid holdout mean within exp_024 publication 95% CI."""
    accuracies = [
        _train_hybrid(
            dataset="breast_cancer",
            seed=seed,
            exp_id=f"real_pub_bc_{seed}",
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
        for seed in VALIDATION_SEEDS
    ]
    mean_acc = sum(accuracies) / len(accuracies)
    lo, hi = EXP_024_HYBRID_CI
    assert lo <= mean_acc <= hi, (
        f"hybrid mean {mean_acc * 100:.2f}% outside publication CI "
        f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
    )


def test_exp_025_hybrid_within_publication_band(tmp_path, monkeypatch):
    """Ten fresh seeds: hybrid holdout mean within exp_025 publication 95% CI."""
    accuracies = [
        _train_hybrid(
            dataset="pima_diabetes",
            seed=seed,
            exp_id=f"real_pub_pima_{seed}",
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
        for seed in VALIDATION_SEEDS
    ]
    mean_acc = sum(accuracies) / len(accuracies)
    lo, hi = EXP_025_HYBRID_CI
    assert lo <= mean_acc <= hi, (
        f"hybrid mean {mean_acc * 100:.2f}% outside publication CI "
        f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
    )
