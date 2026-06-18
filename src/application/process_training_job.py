"""Use case: run training for an existing job and persist the outcome."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import UTC, datetime

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import TrainNanomodelError
from src.application.train_nanomodel import execute as train_execute
from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.domain.repositories.training_job_repository import TrainingJobRepository
from src.shared.result import Fail, Ok, Result, ok


class ProcessTrainingJobError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(UTC)


@contextmanager
def _device_context(device: str):
    previous = os.environ.get("QML_DEVICE")
    os.environ["QML_DEVICE"] = device
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("QML_DEVICE", None)
        else:
            os.environ["QML_DEVICE"] = previous


def execute(
    job: TrainingJob,
    repo: TrainingJobRepository,
    *,
    save_checkpoints: bool = False,
) -> Result[TrainingJob, ProcessTrainingJobError]:
    """Transition job to RUNNING, execute training, and persist terminal status."""
    if job.status not in {TrainingJobStatus.PENDING, TrainingJobStatus.RUNNING}:
        return ok(job)

    job.status = TrainingJobStatus.RUNNING
    job.updated_at = _now()
    job.version += 1
    repo.save(job)

    train_dto = TrainNanomodelDTO(
        model_name=job.model_name,
        dataset=job.dataset,
        profile=job.profile,
        epochs=job.epochs,
        seed=job.seed,
        exp_id=job.exp_id,
        save_checkpoints=save_checkpoints,
    )

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    with _device_context(job.device):
        outcome = train_execute(train_dto)

    if isinstance(outcome, Fail):
        err: TrainNanomodelError = outcome.error
        job.status = TrainingJobStatus.FAILED
        job.error_code = err.code
        job.error_message = err.message
        job.updated_at = _now()
        job.version += 1
        repo.save(job)
        return ok(job)

    assert isinstance(outcome, Ok)
    result = outcome.value
    job.status = TrainingJobStatus.COMPLETED
    job.result = {
        "exp_id": result.exp_id,
        "accuracy": result.accuracy,
        "loss": result.loss,
        "elapsed_s": result.elapsed_s,
        "n_params": result.n_params,
        "n_epochs": result.n_epochs,
        "seed": result.seed,
        "device": job.device,
        "checkpoint_path": result.checkpoint_path,
    }
    job.updated_at = _now()
    job.version += 1
    repo.save(job)
    return ok(job)
