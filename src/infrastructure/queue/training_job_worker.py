"""Background worker for async training jobs."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from src.application.process_training_job import execute as process_job
from src.infrastructure.database.connection import connect
from src.infrastructure.database.repositories.sqlite_training_job_repository import (
    SqliteTrainingJobRepository,
)


class TrainingJobWorker:
    """Poll SQLite for PENDING jobs and process them on a daemon thread."""

    def __init__(self, db_path: Path, *, poll_interval_s: float = 0.25) -> None:
        self._db_path = db_path
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="training-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            conn = connect(self._db_path)
            try:
                repo = SqliteTrainingJobRepository(conn)
                job = repo.claim_next_pending()
                if job is None:
                    time.sleep(self._poll_interval_s)
                    continue
                process_job(job, repo)
            finally:
                conn.close()
