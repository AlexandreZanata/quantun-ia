"""E2E tests for JWT auth and async training job queue."""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from src.infrastructure.auth import jwt_keys
from src.presentation.http.app import create_app


def _generate_rsa_pem_pair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


@pytest.fixture
def secured_client(tmp_path, monkeypatch):
    db_path = tmp_path / "api-secured.db"
    private_pem, public_pem = _generate_rsa_pem_pair()
    jwt_keys.load_signing_key.cache_clear()
    jwt_keys.load_public_key.cache_clear()
    monkeypatch.setenv("DATABASE_PATH", str(db_path))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("API_AUTH_REQUIRED", "1")
    monkeypatch.setenv("API_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("JWT_PRIVATE_KEY_PEM", private_pem)
    monkeypatch.setenv("JWT_PUBLIC_KEY_PEM", public_pem)
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    log_file = tmp_path / "experiments.jsonl"
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", log_file)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _issue_token(client: TestClient, tenant_id: str = "tenant-secure") -> str:
    res = client.post(
        "/api/v1/auth/token",
        json={"tenant_id": tenant_id, "api_key": "test-secret"},
    )
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


def test_auth_token_issue_and_refresh(secured_client):
    token_res = secured_client.post(
        "/api/v1/auth/token",
        json={"tenant_id": "tenant-secure", "api_key": "test-secret"},
    )
    assert token_res.status_code == 200
    body = token_res.json()
    assert body["token_type"] == "Bearer"
    assert body["access_token"]
    assert body["refresh_token"]

    refresh_res = secured_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": body["refresh_token"]},
    )
    assert refresh_res.status_code == 200
    refreshed = refresh_res.json()
    assert refreshed["access_token"]
    assert refreshed["refresh_token"] != body["refresh_token"]


def test_training_job_requires_bearer_when_auth_enabled(secured_client):
    res = secured_client.post(
        "/api/v1/training-jobs",
        json={"model_name": "perceptron", "dataset": "breast_cancer", "profile": "ci", "epochs": 2},
        headers={"X-Tenant-ID": "tenant-secure"},
    )
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "UNAUTHORIZED"


def test_jwt_authenticated_sync_job_completes(secured_client):
    access_token = _issue_token(secured_client)
    res = secured_client.post(
        "/api/v1/training-jobs",
        json={"model_name": "perceptron", "dataset": "breast_cancer", "profile": "ci", "epochs": 3},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["status"] == "COMPLETED"
    assert body["tenant_id"] == "tenant-secure"
    assert body["result"]["accuracy"] >= 0.5


def test_async_job_returns_pending_then_completes(secured_client):
    access_token = _issue_token(secured_client)
    res = secured_client.post(
        "/api/v1/training-jobs",
        json={
            "model_name": "perceptron",
            "dataset": "breast_cancer",
            "profile": "ci",
            "epochs": 3,
            "async_mode": True,
            "device": "cpu",
        },
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert res.status_code == 202, res.text
    body = res.json()
    assert body["status"] == "PENDING"
    job_id = body["id"]

    deadline = time.time() + 30
    final_status = body["status"]
    while time.time() < deadline:
        poll = secured_client.get(
            f"/api/v1/training-jobs/{job_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert poll.status_code == 200
        final_status = poll.json()["status"]
        if final_status in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.2)

    assert final_status == "COMPLETED"
    final = secured_client.get(
        f"/api/v1/training-jobs/{job_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    ).json()
    assert final["result"]["accuracy"] >= 0.5
    assert final["device"] == "cpu"


def test_invalid_api_key_returns_401(secured_client):
    res = secured_client.post(
        "/api/v1/auth/token",
        json={"tenant_id": "tenant-secure", "api_key": "wrong-key"},
    )
    assert res.status_code == 401
    assert res.json()["detail"]["code"] == "INVALID_CREDENTIALS"
