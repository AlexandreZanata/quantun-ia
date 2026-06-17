"""
EXP 020 — REST API smoke: training job via HTTP matches Nano Trainer path.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from src.presentation.http.app import create_app

EXP_ID = "exp_020"
ACC_MIN = 0.35
ACC_MAX = 1.0


def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    db_path = Path("data") / "exp_020_api.db"
    os.environ["DATABASE_PATH"] = str(db_path)

    app = create_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text

        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text

        res = client.post(
            "/api/v1/training-jobs",
            json={
                "model_name": "perceptron",
                "dataset": "breast_cancer",
                "profile": "ci",
                "epochs": 3,
            },
            headers={"X-Tenant-ID": "local"},
        )
        assert res.status_code == 201, res.text
        body = res.json()
        assert body["status"] == "COMPLETED", body
        acc = body["result"]["accuracy"]
        assert ACC_MIN <= acc <= ACC_MAX, f"accuracy {acc} out of bounds"

        job_id = body["id"]
        fetched = client.get(
            f"/api/v1/training-jobs/{job_id}",
            headers={"X-Tenant-ID": "local"},
        )
        assert fetched.status_code == 200
        assert fetched.json()["id"] == job_id

    print(f"exp_020 complete — API holdout accuracy {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
