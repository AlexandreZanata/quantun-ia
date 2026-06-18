"""Unit tests for open dataset serve artifact publishing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.application.open_serve import (
    load_open_holdout_rows,
    open_dataset_feature_count,
    publish_large_nano_serve_artifact,
    verify_serve_artifact,
)
from src.classical.large_nano_mlp import LargeNanoMLP
from src.training.checkpoints import checkpoint_path, save_checkpoint


def _write_higgs_bundle(root: Path, n_train: int = 200, n_val: int = 50, n_test: int = 80) -> None:
    n_features = 28
    out = root / "data" / "open" / "higgs" / "processed" / "v1"
    out.mkdir(parents=True, exist_ok=True)

    for name, rows in (("train", n_train), ("val", n_val), ("test", n_test)):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(rows, n_features)).astype(np.float32),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        frame["label"] = (np.arange(rows) % 2).astype(np.int32)
        frame.to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        "datasets": [
            {
                "id": "higgs_v1",
                "path": "higgs/processed/v1",
                "ready": True,
                "n_features": 28,
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


def _write_training_checkpoint(root: Path, *, exp_id: str, seed: int, n_features: int = 28) -> None:
    source_dir = checkpoint_path(exp_id, "large_nano_mlp", seed)
    model = LargeNanoMLP(input_dim=n_features)
    save_checkpoint(
        model,
        source_dir,
        config={"input_dim": n_features},
        metadata={"source": "unit_test"},
    )


def test_open_dataset_feature_count_higgs():
    assert open_dataset_feature_count("higgs_v1") == 28


def test_load_open_holdout_rows_subsample(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_higgs_bundle(tmp_path)
    rows = load_open_holdout_rows("higgs_v1", tmp_path, n_rows=20, random_state=42)
    assert len(rows) == 20
    assert all(len(row) == 28 for row in rows)


def test_publish_large_nano_serve_artifact(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("src.training.checkpoints.ARTIFACTS_ROOT", tmp_path / "artifacts")
    _write_higgs_bundle(tmp_path)
    _write_training_checkpoint(tmp_path, exp_id="exp_032", seed=42)

    target = publish_large_nano_serve_artifact(
        tmp_path,
        exp_id="exp_032",
        model_name="large_nano_mlp",
        dataset_id="higgs_v1",
        seed=42,
    )
    assert (target / "best.pt").is_file()
    assert (target / "scaler.joblib").is_file()
    verify_serve_artifact("exp_032", "large_nano_mlp", "higgs_v1", seed=42)

    state = torch.load(target / "best.pt", map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
