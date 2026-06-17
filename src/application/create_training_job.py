"""Use case: create a training job (sync or async enqueue)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from src.application.process_training_job import execute as process_job
from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.domain.repositories.training_job_repository import TrainingJobRepository
from src.shared.result import Ok, Result, fail, ok


@dataclass(frozen=True)
class CreateTrainingJobDTO:
    tenant_id: str
    model_name: str
    dataset: str
    profile: str = "mini"
    epochs: int | None = None
    seed: int | None = None
    exp_id: str = "nano_train"
    device: str = "auto"
    async_mode: bool = False


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
    """Persist a job; run synchronously or leave PENDING for the worker."""
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
        device=dto.device,
        status=TrainingJobStatus.PENDING,
        created_at=now,
        updated_at=now,
        version=1,
    )
    repo.save(job)

    if dto.async_mode:
        return ok(job)

    outcome = process_job(job, repo)
    assert isinstance(outcome, Ok)
    return ok(outcome.value)
