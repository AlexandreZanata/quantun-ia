"""E2E — agro maize predict and model card routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.presentation.http.app import create_app

ROOT = Path(__file__).resolve().parents[2]
ACYD_CKPT = (
    ROOT
    / "artifacts"
    / "exp_092"
    / "residual_nano_distill_acyd_maize_brazil_v1"
    / "seed_42"
    / "best.pt"
)


@pytest.mark.e2e
def test_agro_maize_predict_and_model_card(tmp_path, monkeypatch):
    if not ACYD_CKPT.is_file():
        pytest.skip("exp_092 distill maize serve checkpoint missing — run make ship-residual-maize")

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cuda")

    app = create_app()
    with TestClient(app) as client:
        card = client.get("/api/v1/models/agro/maize/card")
        assert card.status_code == 200
        assert card.json()["model_id"] == "residual_nano_distill_acyd_maize"

        predict = client.post(
            "/api/v1/predict/agro/maize",
            json={
                "municipality": "Sorriso",
                "state": "MT",
                "crop_year": 2020,
                "latitude": -12.54,
                "longitude": -55.71,
                "area_harvested_ha": 30000,
                "precip_mean": 5.0,
                "tmax_peak_k": 305.0,
                "ndvi_mean": 3.5,
            },
            headers={"X-Tenant-ID": "tenant-agro"},
        )
        assert predict.status_code == 200
        body = predict.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["risk_tier"] in {"low", "moderate", "high"}
        assert body["model_id"] == "residual_nano_distill_acyd_maize"
        assert len(body["top_drivers"]) == 3
