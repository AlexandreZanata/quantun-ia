"""Unit tests for async job claim and refresh token rotation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.application.refresh_tokens import RefreshTokensDTO
from src.application.refresh_tokens import execute as refresh_execute
from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.infrastructure.database.connection import connect, init_schema
from src.infrastructure.database.repositories.sqlite_refresh_token_repository import (
    SqliteRefreshTokenRepository,
)
from src.infrastructure.database.repositories.sqlite_training_job_repository import (
    SqliteTrainingJobRepository,
)
from src.shared.result import Ok


@pytest.fixture
def conn(tmp_path):
    connection = connect(tmp_path / "queue.db")
    init_schema(connection)
    yield connection
    connection.close()


def _job(job_id: str, *, status: TrainingJobStatus = TrainingJobStatus.PENDING) -> TrainingJob:
    now = datetime.now(UTC)
    return TrainingJob(
        id=job_id,
        tenant_id="tenant-a",
        model_name="perceptron",
        dataset="breast_cancer",
        profile="ci",
        status=status,
        device="cpu",
        created_at=now,
        updated_at=now,
        version=1,
    )


def test_claim_next_pending_marks_running(conn):
    repo = SqliteTrainingJobRepository(conn)
    repo.save(_job("job-1"))
    claimed = repo.claim_next_pending()
    assert claimed is not None
    assert claimed.status == TrainingJobStatus.RUNNING
    assert claimed.version == 2
    assert repo.claim_next_pending() is None


def test_refresh_token_rotation_revokes_old_token(conn):
    refresh_repo = SqliteRefreshTokenRepository(conn)
    refresh_repo.save(
        tenant_id="tenant-a",
        user_id="user-1",
        token="refresh-old",
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    outcome = refresh_execute(RefreshTokensDTO(refresh_token="refresh-old"), refresh_repo)
    assert isinstance(outcome, Ok)
    assert refresh_repo.find_active("refresh-old") is None
    assert refresh_repo.find_active(outcome.value.refresh_token) is not None
