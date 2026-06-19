"""E2E — agro soybean predict and model card routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.presentation.http.app import create_app

ROOT = Path(__file__).resolve().parents[2]
ACYD_CKPT = ROOT / "artifacts" / "exp_060" / "large_nano_mlp_acyd_soy_brazil_v1" / "seed_42" / "best.pt"


@pytest.mark.e2e
def test_agro_soy_predict_and_model_card(tmp_path, monkeypatch):
    if not ACYD_CKPT.is_file():
        pytest.skip("exp_060 ACYD checkpoint missing")

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cuda")

    app = create_app()
    with TestClient(app) as client:
        card = client.get("/api/v1/models/agro/soy/card")
        assert card.status_code == 200
        assert card.json()["model_id"] == "large_nano_mlp_acyd_soy"

        predict = client.post(
            "/api/v1/predict/agro/soy",
            json={
                "municipality": "Lucas do Rio Verde",
                "state": "MT",
                "crop_year": 2020,
                "latitude": -13.06,
                "longitude": -55.91,
                "area_harvested_ha": 25000,
                "precip_mean": 5.5,
                "tmax_peak_k": 304.0,
                "ndvi_mean": 3.8,
            },
            headers={"X-Tenant-ID": "tenant-agro"},
        )
        assert predict.status_code == 200
        body = predict.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk_tier"] in {"low", "moderate", "high"}
        assert len(body["top_drivers"]) == 3
