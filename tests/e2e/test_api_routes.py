"""E2E tests for REST API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.presentation.http.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api.db"
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_ready_returns_200(client):
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_metrics_prometheus_format(client):
    client.get("/health")
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "http_requests_total" in res.text


def test_post_training_job_creates_completed_job(client):
    res = client.post(
        "/api/v1/training-jobs",
        json={
            "model_name": "perceptron",
            "dataset": "breast_cancer",
            "profile": "ci",
            "epochs": 3,
        },
        headers={"X-Tenant-ID": "tenant-test"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "COMPLETED"
    assert body["tenant_id"] == "tenant-test"
    assert body["result"]["accuracy"] >= 0.5

    get_res = client.get(
        f"/api/v1/training-jobs/{body['id']}",
        headers={"X-Tenant-ID": "tenant-test"},
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == body["id"]


def test_get_training_job_wrong_tenant_returns_404(client):
    res = client.post(
        "/api/v1/training-jobs",
        json={"model_name": "perceptron", "dataset": "breast_cancer", "profile": "ci", "epochs": 2},
        headers={"X-Tenant-ID": "tenant-a"},
    )
    job_id = res.json()["id"]
    missing = client.get(
        f"/api/v1/training-jobs/{job_id}",
        headers={"X-Tenant-ID": "tenant-b"},
    )
    assert missing.status_code == 404


def test_post_invalid_pair_returns_failed_job(client):
    res = client.post(
        "/api/v1/training-jobs",
        json={"model_name": "perceptron", "dataset": "sequential_phase", "profile": "ci"},
    )
    assert res.status_code == 201
    assert res.json()["status"] == "FAILED"
    assert res.json()["error_code"] == "INVALID_PAIR"


def test_leaderboard_endpoint_returns_json(client):
    res = client.get("/api/v1/benchmarks/leaderboard")
    assert res.status_code == 200
    assert "rows" in res.json()


def test_pwa_index_served(client):
    res = client.get("/pwa/")
    assert res.status_code == 200
    assert "QML BENCHMARKS" in res.text
