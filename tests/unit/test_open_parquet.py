"""Unit tests for open parquet loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.open_parquet import load_open_parquet_splits


def _write_higgs_bundle(root: Path, n_train: int = 200, n_val: int = 50, n_test: int = 50) -> None:
    n_features = 28
    out = root / "data" / "open" / "higgs" / "processed" / "v1"
    out.mkdir(parents=True, exist_ok=True)

    for name, rows in (("train", n_train), ("val", n_val), ("test", n_test)):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(rows, n_features)).astype(np.float32),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        frame["label"] = np.zeros(rows, dtype=np.int32)
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
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")


def test_load_open_parquet_splits_scales_train_only(tmp_path: Path):
    _write_higgs_bundle(tmp_path)
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = load_open_parquet_splits(
        "higgs_v1",
        tmp_path,
        n_train_rows=100,
        n_val_rows=25,
    )
    assert x_train.shape == (100, 28)
    assert x_val.shape == (25, 28)
    assert x_test.shape == (50, 28)
    assert abs(float(x_train.mean())) < 0.2
    assert scaler.mean_.shape == (28,)
