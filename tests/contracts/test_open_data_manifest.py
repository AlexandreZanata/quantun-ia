"""Contract tests for Phase L open data manifest and parquet exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.open_manifest import (
    dvc_pointer_path,
    expected_feature_columns,
    validate_open_data,
    verify_stratified_balance,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "open" / "manifest.json"
SCHEMA_PATH = ROOT / "data" / "open" / "schemas" / "tabular_binary_v1.json"

EXPECTED_HIGGS_COUNTS = {
    "total": 1_150_000,
    "train": 805_000,
    "val": 172_500,
    "test": 172_500,
}

EXPECTED_SYNTHEA_COUNTS = {
    "total": 1_000_000,
    "train": 700_000,
    "val": 150_000,
    "test": 150_000,
}

EXPECTED_NIHR_COUNTS = {
    "total": 100_000,
    "train": 70_000,
    "val": 15_000,
    "test": 15_000,
}

EXPECTED_GOBUG_COUNTS = {
    "total": 38_818,
    "train": 27_172,
    "val": 5_822,
    "test": 5_824,
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


def test_manifest_synthea_cv_risk_v1_metadata():
    manifest = _load_manifest()
    synthea = _dataset_by_id(manifest, "synthea_cv_risk_v1")
    assert synthea["license"] == "MIT"
    assert synthea["n_features"] == 40
    assert synthea["build_script"] == "scripts/build_synthea_cv_risk.py"
    assert Path(ROOT / synthea["build_script"]).is_file()
    assert synthea["row_counts"] == EXPECTED_SYNTHEA_COUNTS


def test_manifest_nihr_cv_synthetic_v1_metadata():
    manifest = _load_manifest()
    nihr = _dataset_by_id(manifest, "nihr_cv_synthetic_v1")
    assert nihr["license"] == "CC0-1.0"
    assert nihr["n_features"] == 13
    assert nihr["build_script"] == "scripts/build_nihr_cv_synthetic.py"
    assert Path(ROOT / nihr["build_script"]).is_file()
    assert nihr["row_counts"] == EXPECTED_NIHR_COUNTS


def test_manifest_code_defects_gobug_v1_metadata():
    manifest = _load_manifest()
    gobug = _dataset_by_id(manifest, "code_defects_gobug_v1")
    assert gobug["n_features"] == 23
    assert gobug["build_script"] == "scripts/build_gobug_subset.py"
    assert Path(ROOT / gobug["build_script"]).is_file()
    assert gobug["row_counts"] == EXPECTED_GOBUG_COUNTS


def test_tabular_binary_schema_file_exists():
    assert SCHEMA_PATH.is_file()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["label_column"]["const"] == "label"
    assert schema["properties"]["dtype"]["const"] == "float32"


def test_higgs_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-higgs" in makefile


def test_synthea_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-synthea-cv" in makefile


def test_nihr_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-nihr-cv" in makefile


def test_gobug_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-gobug" in makefile


def test_acyd_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-acyd-soy" in makefile
    assert "data-open-acyd-maize" in makefile


def test_cybench_build_script_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-cybench-maize" in makefile


def test_manifest_cybench_maize_us_v1_metadata():
    manifest = _load_manifest()
    ds = _dataset_by_id(manifest, "cybench_maize_us_v1")
    assert ds["n_features"] == 26
    assert ds["license"] == "EUPL-1.2"
    assert ds["build_script"] == "scripts/build_open_cybench_maize.py"
    assert Path(ROOT / ds["build_script"]).is_file()
    assert ds["ready"] is True
    assert ds["row_counts"]["train"] == 7170
    assert ds["row_counts"]["val"] == 2185
    assert ds["row_counts"]["test"] == 1952


def test_manifest_acyd_soy_brazil_v1_metadata():
    manifest = _load_manifest()
    acyd = _dataset_by_id(manifest, "acyd_soy_brazil_v1")
    assert acyd["n_features"] == 37
    assert acyd["build_script"] == "scripts/build_open_acyd_soy.py"
    assert Path(ROOT / acyd["build_script"]).is_file()
    assert Path(ROOT / "scripts/download_acyd_brazil.py").is_file()
    assert acyd["ready"] is True
    assert acyd["row_counts"]["train"] == 50_107


def test_manifest_acyd_maize_brazil_v1_metadata():
    manifest = _load_manifest()
    acyd = _dataset_by_id(manifest, "acyd_maize_brazil_v1")
    assert acyd["n_features"] == 37
    assert acyd["build_script"] == "scripts/build_open_acyd_maize.py"
    assert Path(ROOT / acyd["build_script"]).is_file()
    assert Path(ROOT / "data/open/acyd_maize_brazil/README.md").is_file()
    assert acyd["ready"] is True
    assert acyd["row_counts"]["train"] == 151_956
    assert acyd["row_counts"]["val"] == 13_566
    assert acyd["row_counts"]["test"] == 13_537


@pytest.mark.parametrize(
    ("dataset_id", "split_name", "expected_rows", "n_features"),
    [
        ("higgs_v1", "train", 805_000, 28),
        ("higgs_v1", "val", 172_500, 28),
        ("higgs_v1", "test", 172_500, 28),
        ("synthea_cv_risk_v1", "train", 700_000, 40),
        ("synthea_cv_risk_v1", "val", 150_000, 40),
        ("synthea_cv_risk_v1", "test", 150_000, 40),
        ("nihr_cv_synthetic_v1", "train", 70_000, 13),
        ("nihr_cv_synthetic_v1", "val", 15_000, 13),
        ("nihr_cv_synthetic_v1", "test", 15_000, 13),
        ("code_defects_gobug_v1", "train", 27_172, 23),
        ("code_defects_gobug_v1", "val", 5_822, 23),
        ("code_defects_gobug_v1", "test", 5_824, 23),
        ("acyd_soy_brazil_v1", "train", 50_107, 37),
        ("acyd_soy_brazil_v1", "val", 5_830, 37),
        ("acyd_soy_brazil_v1", "test", 5_856, 37),
        ("acyd_maize_brazil_v1", "train", 151_956, 37),
        ("acyd_maize_brazil_v1", "val", 13_566, 37),
        ("acyd_maize_brazil_v1", "test", 13_537, 37),
    ],
)
def test_open_dataset_splits_when_ready(
    dataset_id: str,
    split_name: str,
    expected_rows: int,
    n_features: int,
):
    manifest = _load_manifest()
    dataset = _dataset_by_id(manifest, dataset_id)
    if not dataset.get("ready"):
        pytest.skip(f"{dataset_id} not built yet (ready=false)")

    processed = ROOT / "data" / "open" / dataset["path"]
    parquet_path = processed / dataset["files"][split_name]
    if not parquet_path.is_file():
        pytest.skip(f"missing processed file: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    assert len(frame) == expected_rows
    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    assert len(feature_cols) == n_features
    assert frame["label"].dtype in (np.int32, "int32", "int64")
    assert frame.isna().sum().sum() == 0
    assert set(frame["label"].unique()).issubset({0, 1})


def _is_image_pack(dataset: dict) -> bool:
    return dataset.get("modality") == "images"


def test_ready_datasets_have_checksums():
    manifest = _load_manifest()
    for dataset in manifest["datasets"]:
        if not dataset.get("ready"):
            continue
        checksums = dataset.get("checksums", {})
        assert checksums, f"{dataset['id']} ready dataset must include checksums"
        processed = ROOT / "data" / "open" / dataset["path"]
        for key, filename in dataset["files"].items():
            assert key in checksums
            assert len(checksums[key]) == 64
            file_path = processed / filename
            if not file_path.is_file():
                pytest.skip(f"open data not on disk (DVC pull required): {file_path}")
            assert file_path.is_file()


def test_ready_datasets_have_dvc_pointer():
    manifest = _load_manifest()
    for dataset in manifest["datasets"]:
        if not dataset.get("ready") or _is_image_pack(dataset):
            continue
        pointer = dvc_pointer_path(ROOT, dataset)
        assert pointer.is_file(), f"missing DVC pointer: {pointer}"


def test_ready_datasets_stratified_balance():
    manifest = _load_manifest()
    for dataset in manifest["datasets"]:
        if not dataset.get("ready") or _is_image_pack(dataset):
            continue
        stats_path = ROOT / "data" / "open" / dataset["path"] / dataset["files"]["stats"]
        if not stats_path.is_file():
            pytest.skip(f"missing stats: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("split_method") in ("temporal_sha_order", "temporal_crop_year"):
            continue
        errors = verify_stratified_balance(stats, tolerance=0.01)
        assert errors == [], f"{dataset['id']}: {errors}"


def test_ready_datasets_tabular_binary_contract():
    manifest = _load_manifest()
    for dataset in manifest["datasets"]:
        if not dataset.get("ready") or _is_image_pack(dataset):
            continue
        processed = ROOT / "data" / "open" / dataset["path"]
        expected_cols = expected_feature_columns(dataset["n_features"]) + ["label"]
        for split_name in ("train", "val", "test"):
            parquet_path = processed / dataset["files"][split_name]
            if not parquet_path.is_file():
                pytest.skip(f"missing processed file: {parquet_path}")
            frame = pd.read_parquet(parquet_path)
            assert list(frame.columns) == expected_cols


def test_open_data_l2_gate_passes_when_built():
    manifest = _load_manifest()
    ready_ids = [
        d["id"]
        for d in manifest["datasets"]
        if d.get("ready") and not _is_image_pack(d)
    ]
    if not ready_ids:
        pytest.skip("no ready tabular datasets")
    for dataset_id in ready_ids:
        dataset = _dataset_by_id(manifest, dataset_id)
        processed = ROOT / "data" / "open" / dataset["path"]
        train_path = processed / dataset["files"]["train"]
        if not train_path.is_file():
            pytest.skip(f"open data not on disk (DVC pull required): {train_path}")
        ok, issues = validate_open_data(ROOT, dataset_id=dataset_id)
        assert ok, f"{dataset_id}: {issues}"


def test_ready_image_packs_have_processed_artifacts():
    """Phase G image packs use stats + split indices, not tabular parquet."""
    manifest = _load_manifest()
    image_packs = [d for d in manifest["datasets"] if d.get("ready") and _is_image_pack(d)]
    assert image_packs, "expected ready image packs (Phase G P0)"
    for dataset in image_packs:
        processed = ROOT / "data" / "open" / dataset["path"]
        stats_path = processed / dataset["files"]["stats"]
        splits_path = processed / dataset["files"]["split_indices"]
        assert stats_path.is_file(), f"missing {stats_path}"
        assert splits_path.is_file(), f"missing {splits_path}"
        assert dataset["row_counts"]["total"] == sum(
            dataset["row_counts"][k] for k in ("train", "val", "test")
        )


def test_data_open_verify_registered_in_makefile():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "data-open-verify" in makefile
