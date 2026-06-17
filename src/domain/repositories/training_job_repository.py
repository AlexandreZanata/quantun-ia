"""TrainingJob repository interface."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.training_job import TrainingJob


class TrainingJobRepository(Protocol):
    def save(self, job: TrainingJob) -> None: ...

    def find_by_id(self, job_id: str, tenant_id: str) -> TrainingJob | None: ...

    def claim_next_pending(self) -> TrainingJob | None: ...
