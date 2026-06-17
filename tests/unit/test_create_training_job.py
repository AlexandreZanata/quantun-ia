"""Unit tests for create_training_job use case."""

from __future__ import annotations

from src.application.create_training_job import CreateTrainingJobDTO, execute
from src.domain.entities.training_job import TrainingJob, TrainingJobStatus
from src.shared.result import Fail, Ok


class InMemoryRepo:
    def __init__(self) -> None:
        self.jobs: dict[str, TrainingJob] = {}

    def save(self, job: TrainingJob) -> None:
        self.jobs[job.id] = job

    def find_by_id(self, job_id: str, tenant_id: str) -> TrainingJob | None:
        job = self.jobs.get(job_id)
        if job is None or job.tenant_id != tenant_id:
            return None
        return job


def test_execute_invalid_tenant_returns_fail():
    repo = InMemoryRepo()
    dto = CreateTrainingJobDTO(tenant_id="", model_name="perceptron", dataset="breast_cancer")
    result = execute(dto, repo)
    assert isinstance(result, Fail)
    assert result.error.code == "INVALID_TENANT"


def test_execute_success_persists_completed_job(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    repo = InMemoryRepo()
    dto = CreateTrainingJobDTO(
        tenant_id="local",
        model_name="perceptron",
        dataset="breast_cancer",
        profile="ci",
        epochs=3,
        seed=42,
    )
    result = execute(dto, repo)
    assert isinstance(result, Ok)
    job = result.value
    assert job.status == TrainingJobStatus.COMPLETED
    assert job.result is not None
    assert 0.5 <= job.result["accuracy"] <= 1.0
    assert repo.find_by_id(job.id, "local") is not None


def test_execute_invalid_pair_persists_failed_job():
    repo = InMemoryRepo()
    dto = CreateTrainingJobDTO(
        tenant_id="local",
        model_name="perceptron",
        dataset="sequential_phase",
        profile="ci",
    )
    result = execute(dto, repo)
    assert isinstance(result, Ok)
    job = result.value
    assert job.status == TrainingJobStatus.FAILED
    assert job.error_code == "INVALID_PAIR"
