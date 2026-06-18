"""Unit tests for open data manifest validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.open_manifest import (
    collect_open_data_issues,
    dvc_pointer_path,
    expected_feature_columns,
    verify_checksums,
    verify_parquet_frame,
    verify_stratified_balance,
)


def _write_minimal_higgs_bundle(root: Path) -> None:
    n_features = 28
    for split, rows in (("train", 100), ("val", 20), ("test", 20)):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(rows, n_features)).astype(np.float32),
            columns=expected_feature_columns(n_features),
        )
        frame["label"] = np.zeros(rows, dtype=np.int32)
        out = root / "data" / "open" / "higgs" / "processed" / "v1"
        out.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out / f"{split}.parquet", index=False)

    stats = {
        "splits": {
            "train": {"positive_rate": 0.53},
            "val": {"positive_rate": 0.529},
            "test": {"positive_rate": 0.531},
        }
    }
    stats_path = root / "data" / "open" / "higgs" / "processed" / "v1" / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    manifest = {
        "datasets": [
            {
                "id": "higgs_v1",
                "path": "higgs/processed/v1",
                "ready": True,
                "n_features": 28,
                "row_counts": {"train": 100, "val": 20, "test": 20},
                "files": {
                    "train": "train.parquet",
                    "val": "val.parquet",
                    "test": "test.parquet",
                    "stats": "stats.json",
                },
                "checksums": {},
            }
        ]
    }
    manifest_path = root / "data" / "open" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dvc_path = root / "data" / "open" / "higgs" / "processed" / "v1.dvc"
    dvc_path.write_text("outs:\n- path: v1\n", encoding="utf-8")


def test_expected_feature_columns_count():
    assert len(expected_feature_columns(28)) == 28
    assert expected_feature_columns(3) == ["feature_0", "feature_1", "feature_2"]


def test_verify_parquet_frame_valid():
    frame = pd.DataFrame(
        {"feature_0": [1.0], "feature_1": [2.0], "label": np.int32([1])},
    )
    assert verify_parquet_frame(frame, n_features=2) == []


def test_verify_parquet_frame_rejects_nan():
    frame = pd.DataFrame(
        {"feature_0": [float("nan")], "label": np.int32([0])},
    )
    errors = verify_parquet_frame(frame, n_features=1)
    assert any("NaN" in e for e in errors)


def test_verify_stratified_balance_within_tolerance():
    stats = {
        "splits": {
            "train": {"positive_rate": 0.530},
            "val": {"positive_rate": 0.529},
            "test": {"positive_rate": 0.531},
        }
    }
    assert verify_stratified_balance(stats, tolerance=0.01) == []


def test_verify_stratified_balance_rejects_drift():
    stats = {
        "splits": {
            "train": {"positive_rate": 0.60},
            "val": {"positive_rate": 0.50},
            "test": {"positive_rate": 0.50},
        }
    }
    errors = verify_stratified_balance(stats, tolerance=0.01)
    assert errors


def test_dvc_pointer_path_derivation():
    dataset = {"path": "higgs/processed/v1"}
    pointer = dvc_pointer_path(Path("/repo"), dataset)
    assert pointer == Path("/repo/data/open/higgs/processed/v1.dvc")


def test_verify_checksums_detects_mismatch(tmp_path: Path):
    out_dir = tmp_path / "v1"
    out_dir.mkdir()
    file_path = out_dir / "train.parquet"
    file_path.write_bytes(b"data")
    dataset = {
        "ready": True,
        "files": {"train": "train.parquet"},
        "checksums": {"train": "0" * 64},
    }
    errors = verify_checksums(dataset, out_dir)
    assert any("checksum mismatch" in e for e in errors)


def test_collect_open_data_issues_on_minimal_bundle(tmp_path: Path):
    _write_minimal_higgs_bundle(tmp_path)
    issues = collect_open_data_issues(tmp_path)
    assert any("missing checksums" in i for i in issues)
