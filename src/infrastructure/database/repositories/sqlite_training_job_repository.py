"""SQLite implementation of TrainingJobRepository."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.domain.repositories.training_job_repository import TrainingJobRepository


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _row_to_entity(row: sqlite3.Row) -> TrainingJob:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return TrainingJob(
        id=row["id"],
        tenant_id=row["tenant_id"],
        model_name=row["model_name"],
        dataset=row["dataset"],
        profile=row["profile"],
        exp_id=row["exp_id"],
        seed=row["seed"],
        epochs=row["epochs"],
        status=TrainingJobStatus(row["status"]),
        result=result,
        error_code=row["error_code"],
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        deleted_at=_parse_dt(row["deleted_at"]),
        version=int(row["version"]),
    )


class SqliteTrainingJobRepository(TrainingJobRepository):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, job: TrainingJob) -> None:
        self._conn.execute(
            """
            INSERT INTO training_jobs (
                id, tenant_id, model_name, dataset, profile, exp_id, seed, epochs,
                status, result_json, error_code, error_message,
                created_at, updated_at, deleted_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                result_json = excluded.result_json,
                error_code = excluded.error_code,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at,
                version = excluded.version
            """,
            (
                job.id,
                job.tenant_id,
                job.model_name,
                job.dataset,
                job.profile,
                job.exp_id,
                job.seed,
                job.epochs,
                job.status.value,
                json.dumps(job.result) if job.result is not None else None,
                job.error_code,
                job.error_message,
                job.created_at.isoformat(),
                job.updated_at.isoformat(),
                job.deleted_at.isoformat() if job.deleted_at else None,
                job.version,
            ),
        )
        self._conn.commit()

    def find_by_id(self, job_id: str, tenant_id: str) -> TrainingJob | None:
        row = self._conn.execute(
            """
            SELECT * FROM training_jobs
            WHERE id = ? AND tenant_id = ? AND deleted_at IS NULL
            """,
            (job_id, tenant_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_entity(row)
