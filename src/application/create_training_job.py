"""Use case: create and run a training job via the Nano Trainer orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import TrainNanomodelError
from src.application.train_nanomodel import execute as train_execute
from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.domain.repositories.training_job_repository import TrainingJobRepository
from src.shared.result import Fail, Ok, Result, fail, ok


@dataclass(frozen=True)
class CreateTrainingJobDTO:
    tenant_id: str
    model_name: str
    dataset: str
    profile: str = "mini"
    epochs: int | None = None
    seed: int | None = None
    exp_id: str = "nano_train"


class CreateTrainingJobError:
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(UTC)


def execute(
    dto: CreateTrainingJobDTO,
    repo: TrainingJobRepository,
) -> Result[TrainingJob, CreateTrainingJobError]:
    """Persist a job, run training synchronously, and update status."""
    if not dto.tenant_id:
        return fail(CreateTrainingJobError("INVALID_TENANT", "tenant_id is required"))

    now = _now()
    job = TrainingJob(
        id=str(uuid.uuid4()),
        tenant_id=dto.tenant_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        profile=dto.profile,
        seed=dto.seed,
        epochs=dto.epochs,
        exp_id=dto.exp_id,
        status=TrainingJobStatus.PENDING,
        created_at=now,
        updated_at=now,
        version=1,
    )
    repo.save(job)

    job.status = TrainingJobStatus.RUNNING
    job.updated_at = _now()
    job.version += 1
    repo.save(job)

    train_dto = TrainNanomodelDTO(
        model_name=dto.model_name,
        dataset=dto.dataset,
        profile=dto.profile,
        epochs=dto.epochs,
        seed=dto.seed,
        exp_id=dto.exp_id,
    )
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
    }
    job.updated_at = _now()
    job.version += 1
    repo.save(job)
    return ok(job)
