"""TrainingJob domain entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class TrainingJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class TrainingJob:
    id: str
    tenant_id: str
    model_name: str
    dataset: str
    profile: str
    status: TrainingJobStatus
    created_at: datetime
    updated_at: datetime
    version: int
    seed: int | None = None
    epochs: int | None = None
    exp_id: str = "nano_train"
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    deleted_at: datetime | None = None
