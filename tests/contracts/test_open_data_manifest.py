"""Contract tests for Phase L open data manifest and parquet exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "open" / "manifest.json"
SCHEMA_PATH = ROOT / "data" / "open" / "schemas" / "tabular_binary_v1.json"

EXPECTED_HIGGS_COUNTS = {
    "total": 1_150_000,
    "train": 805_000,
    "val": 172_500,
    "test": 172_500,
}


def _load_manifest() -> dict:
    assert MANIFEST_PATH.is_file(), "data/open/manifest.json must exist"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _dataset_by_id(manifest: dict, dataset_id: str) -> dict:
    for item in manifest.get("datasets", []):
        if item.get("id") == dataset_id:
            return item
    raise KeyError(dataset_id)


def test_manifest_schema_version_and_phase():
    manifest = _load_manifest()
    assert manifest["schema_version"] == "1"
    assert manifest["phase"] == "L"
    assert "datasets" in manifest
    assert len(manifest["datasets"]) >= 1


def test_manifest_higgs_v1_metadata():
    manifest = _load_manifest()
    higgs = _dataset_by_id(manifest, "higgs_v1")
    assert higgs["license"] == "CC0-1.0"
    assert higgs["n_features"] == 28
    assert higgs["build_script"] == "scripts/build_open_higgs.py"
    assert Path(ROOT / higgs["build_script"]).is_file()
    assert higgs["row_counts"] == EXPECTED_HIGGS_COUNTS


def test_tabular_binary_schema_file_exists():
    assert SCHEMA_PATH.is_file()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["label_column"]["const"] == "label"
    assert schema["properties"]["dtype"]["const"] == "float32"


def test_higgs_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-higgs" in makefile


@pytest.mark.parametrize(
    ("split_name", "expected_rows"),
    [
        ("train", 805_000),
        ("val", 172_500),
        ("test", 172_500),
    ],
)
def test_higgs_processed_splits_when_ready(split_name: str, expected_rows: int):
    manifest = _load_manifest()
    higgs = _dataset_by_id(manifest, "higgs_v1")
    if not higgs.get("ready"):
        pytest.skip("higgs_v1 not built yet (ready=false)")

    processed = ROOT / "data" / "open" / higgs["path"]
    parquet_name = higgs["files"][split_name]
    parquet_path = processed / parquet_name
    if not parquet_path.is_file():
        pytest.skip(f"missing processed file: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    assert len(frame) == expected_rows
    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    assert len(feature_cols) == higgs["n_features"]
    assert frame["label"].dtype in (np.int32, "int32", "int64")
    assert frame.isna().sum().sum() == 0
    assert set(frame["label"].unique()).issubset({0, 1})


def test_higgs_checksums_when_ready():
    manifest = _load_manifest()
    higgs = _dataset_by_id(manifest, "higgs_v1")
    if not higgs.get("ready"):
        pytest.skip("higgs_v1 not built yet (ready=false)")

    checksums = higgs.get("checksums", {})
    assert checksums, "ready dataset must include checksums"
    processed = ROOT / "data" / "open" / higgs["path"]
    for key, filename in higgs["files"].items():
        assert key in checksums
        assert len(checksums[key]) == 64
        assert (processed / filename).is_file()
