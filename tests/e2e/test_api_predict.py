"""E2E — train job with checkpoint, then POST /api/v1/predictions."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.presentation.http.app import create_app


def test_train_and_predict_breast_cancer(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.metrics.LOGS_PATH", tmp_path / "experiments.jsonl")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")

    app = create_app()
    with TestClient(app) as client:
        train = client.post(
            "/api/v1/training-jobs",
            json={
                "model_name": "hybrid_sandwich",
                "dataset": "breast_cancer",
                "profile": "ci",
                "epochs": 6,
                "seed": 7,
                "exp_id": "api_predict_demo",
                "save_checkpoints": True,
            },
            headers={"X-Tenant-ID": "tenant-predict"},
        )
        assert train.status_code == 201
        assert train.json()["status"] == "COMPLETED"

        predict = client.post(
            "/api/v1/predictions",
            json={
                "exp_id": "api_predict_demo",
                "model_name": "hybrid_sandwich",
                "dataset": "breast_cancer",
                "seed": 7,
                "features": [[0.0] * 30, [1.0] * 30],
            },
            headers={"X-Tenant-ID": "tenant-predict"},
        )
        assert predict.status_code == 200
        body = predict.json()
        assert len(body["probabilities"]) == 2
        assert len(body["labels"]) == 2
        assert body["checkpoint_path"]
