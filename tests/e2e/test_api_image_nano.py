"""E2E — image predict and model card routes (Phase K)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.application.image_nano_ship import is_nano_unet_shipped
from src.presentation.http.app import create_app

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_image_predict_and_model_card(tmp_path, monkeypatch):
    if not is_nano_unet_shipped(ROOT):
        pytest.skip("nano_unet_cifar missing — run make ship-nano-unet-cifar")

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    monkeypatch.setenv("QML_DEVICE", "cuda")

    app = create_app()
    with TestClient(app) as client:
        card = client.get("/api/v1/models/image/card")
        assert card.status_code == 200
        assert card.json()["registry_key"] == "nano_unet_cifar"
        assert card.json()["ready"] is True

        predict = client.post(
            "/api/v1/predict/image",
            json={"n": 2, "seed": 0, "mode": "i2i"},
            headers={"X-Tenant-ID": "tenant-image"},
        )
        assert predict.status_code == 200
        body = predict.json()
        assert body["model_key"] == "nano_unet_cifar"
        assert body["n"] == 2
        assert len(body["png_base64"]) == 2
        assert body["img_size"] == 32
