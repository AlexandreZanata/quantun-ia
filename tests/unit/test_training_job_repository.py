"""Unit tests for SQLite training job repository."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.infrastructure.database.connection import connect, init_schema
from src.infrastructure.database.repositories.sqlite_training_job_repository import (
    SqliteTrainingJobRepository,
)


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    yield SqliteTrainingJobRepository(conn)
    conn.close()


def _sample_job(**kwargs) -> TrainingJob:
    now = datetime.now(UTC)
    defaults = dict(
        id="job-1",
        tenant_id="tenant-a",
        model_name="perceptron",
        dataset="breast_cancer",
        profile="ci",
        status=TrainingJobStatus.PENDING,
        created_at=now,
        updated_at=now,
        version=1,
    )
    defaults.update(kwargs)
    return TrainingJob(**defaults)


def test_save_and_find_by_id(repo):
    job = _sample_job()
    repo.save(job)
    found = repo.find_by_id("job-1", "tenant-a")
    assert found is not None
    assert found.model_name == "perceptron"
    assert found.status == TrainingJobStatus.PENDING


def test_find_by_id_filters_tenant(repo):
    repo.save(_sample_job())
    assert repo.find_by_id("job-1", "tenant-b") is None


def test_update_status_and_result(repo):
    job = _sample_job()
    repo.save(job)
    job.status = TrainingJobStatus.COMPLETED
    job.result = {"accuracy": 0.91}
    job.version = 2
    repo.save(job)
    found = repo.find_by_id("job-1", "tenant-a")
    assert found is not None
    assert found.status == TrainingJobStatus.COMPLETED
    assert found.result == {"accuracy": 0.91}
    assert found.version == 2
