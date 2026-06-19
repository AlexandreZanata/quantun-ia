"""Unit tests for nanomodel ship and export pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from src.application.model_export import (
    export_bundle,
    install_bundle_to_artifacts,
    write_manifest_sha256,
)
from src.application.nanomodel_registry import get_nanomodel_spec
from src.application.nanomodel_ship import ShipNanomodelDTO, execute
from src.application.open_serve import publish_large_nano_serve_artifact
from src.classical.large_nano_mlp import LargeNanoMLP
from src.shared.result import Ok
from src.training.checkpoints import checkpoint_path, save_checkpoint, save_scaler
from tests.unit.test_open_serve import _write_higgs_bundle


def _write_synthea_bundle(root: Path, n_features: int = 40) -> None:
    import numpy as np
    import pandas as pd

    out = root / "data" / "open" / "synthea_cv_risk" / "processed" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", 200), ("val", 50), ("test", 80)):
        frame = pd.DataFrame(
            np.random.default_rng(1).normal(size=(rows, n_features)).astype(np.float32),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        frame["label"] = (np.arange(rows) % 2).astype(np.int32)
        frame.to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        "datasets": [
            {
                "id": "synthea_cv_risk_v1",
                "path": "synthea_cv_risk/processed/v1",
                "ready": True,
                "n_features": n_features,
                "files": {
                    "train": "train.parquet",
                    "val": "val.parquet",
                    "test": "test.parquet",
                    "stats": "stats.json",
                },
            }
        ]
    }
    manifest_path = root / "data" / "open" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_checkpoint(root: Path, *, exp_id: str, n_features: int) -> None:
    source_dir = checkpoint_path(exp_id, "large_nano_mlp", 42)
    model = LargeNanoMLP(input_dim=n_features)
    save_checkpoint(
        model,
        source_dir,
        config={"input_dim": n_features},
        metadata={"source": "unit_test"},
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit([[0.0] * n_features])
    save_scaler(scaler, source_dir)


def test_export_bundle_writes_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_synthea_bundle(tmp_path)
    _write_checkpoint(tmp_path, exp_id="exp_034", n_features=40)

    serve_dir = publish_large_nano_serve_artifact(
        tmp_path,
        exp_id="exp_034",
        model_name="large_nano_mlp",
        dataset_id="synthea_cv_risk_v1",
        seed=42,
    )
    spec = get_nanomodel_spec("large_nano_mlp_synthea")
    bundle_dir = export_bundle(spec, serve_dir, root=tmp_path, stages={"test": True})

    assert (bundle_dir / "best.pt").is_file()
    assert (bundle_dir / "MANIFEST.sha256").is_file()
    assert (bundle_dir / "inference" / "predict.py").is_file()
    assert (bundle_dir / "metrics.json").is_file()


def test_install_bundle_to_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_synthea_bundle(tmp_path)
    _write_checkpoint(tmp_path, exp_id="exp_034", n_features=40)
    serve_dir = publish_large_nano_serve_artifact(
        tmp_path,
        exp_id="exp_034",
        model_name="large_nano_mlp",
        dataset_id="synthea_cv_risk_v1",
        seed=42,
    )
    spec = get_nanomodel_spec("large_nano_mlp_synthea")
    bundle_dir = export_bundle(spec, serve_dir, root=tmp_path, stages={})
    target = install_bundle_to_artifacts(spec, bundle_dir)
    assert (target / "best.pt").is_file()
    assert (target / "scaler.joblib").is_file()


def test_ship_skip_train_skip_gate(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_synthea_bundle(tmp_path)
    _write_checkpoint(tmp_path, exp_id="exp_034", n_features=40)

    outcome = execute(
        ShipNanomodelDTO(
            registry_key="large_nano_mlp_synthea",
            root=tmp_path,
            profile="ci",
            skip_train=True,
            skip_gate=True,
        )
    )
    assert isinstance(outcome, Ok)
    bundle = Path(outcome.value.bundle_dir)
    assert bundle.is_dir()
    write_manifest_sha256(bundle)


def test_ship_higgs_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QML_DEVICE", "cpu")
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_higgs_bundle(tmp_path)
    _write_checkpoint(tmp_path, exp_id="exp_032", n_features=28)

    outcome = execute(
        ShipNanomodelDTO(
            registry_key="large_nano_mlp_higgs",
            root=tmp_path,
            profile="ci",
            skip_train=True,
            skip_gate=True,
        )
    )
    assert isinstance(outcome, Ok)
    onnx_path = Path(outcome.value.bundle_dir) / "exports" / "onnx" / "model.onnx"
    ts_path = Path(outcome.value.bundle_dir) / "exports" / "torchscript" / "model.pt"
    assert onnx_path.is_file() or ts_path.is_file()
