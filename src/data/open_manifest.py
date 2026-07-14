"""Open dataset manifest validation — Phase L2 schema + checksum gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.open_higgs import sha256_file

LABEL_COLUMN = "label"
STRATIFIED_TOLERANCE = 0.01


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and parse data/open/manifest.json."""
    if not path.is_file():
        msg = f"manifest not found: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def get_dataset(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Return a dataset entry by id."""
    for item in manifest.get("datasets", []):
        if item.get("id") == dataset_id:
            return item
    msg = f"dataset not found in manifest: {dataset_id}"
    raise KeyError(msg)


def processed_dir(root: Path, dataset: dict[str, Any]) -> Path:
    """Return absolute path to a dataset processed directory."""
    return root / "data" / "open" / dataset["path"]


def dvc_pointer_path(root: Path, dataset: dict[str, Any]) -> Path:
    """Return expected DVC pointer path for a processed dataset directory."""
    rel = Path("data") / "open" / dataset["path"]
    return root / rel.parent / f"{rel.name}.dvc"


def expected_feature_columns(n_features: int) -> list[str]:
    """Canonical feature column names for tabular_binary_v1."""
    return [f"feature_{i}" for i in range(n_features)]


def verify_parquet_frame(frame: pd.DataFrame, n_features: int) -> list[str]:
    """Validate a parquet frame against tabular_binary_v1 contract."""
    errors: list[str] = []
    expected_cols = expected_feature_columns(n_features) + [LABEL_COLUMN]
    if list(frame.columns) != expected_cols:
        errors.append(f"columns mismatch: expected {expected_cols[:2]}…{expected_cols[-1]}")
    if frame.isna().sum().sum() > 0:
        errors.append("NaN values forbidden after export")
    label_values = set(frame[LABEL_COLUMN].unique().tolist())
    if not label_values <= {0, 1}:
        errors.append(f"label must be {{0, 1}}, got {sorted(label_values)}")
    for col in expected_feature_columns(n_features):
        if frame[col].dtype not in (float, "float32", "float64"):
            errors.append(f"{col} must be float32, got {frame[col].dtype}")
    return errors


def verify_checksums(dataset: dict[str, Any], out_dir: Path) -> list[str]:
    """Verify manifest SHA-256 checksums match on-disk files."""
    errors: list[str] = []
    checksums = dataset.get("checksums", {})
    if dataset.get("ready") and not checksums:
        errors.append("ready dataset missing checksums")
        return errors
    for key, filename in dataset.get("files", {}).items():
        if key not in checksums:
            if dataset.get("ready"):
                errors.append(f"missing checksum for {key}")
            continue
        file_path = out_dir / filename
        if not file_path.is_file():
            errors.append(f"missing file: {file_path}")
            continue
        actual = sha256_file(file_path)
        if actual != checksums[key]:
            errors.append(f"checksum mismatch for {filename}")
    return errors


def verify_stratified_balance(stats: dict[str, Any], tolerance: float = STRATIFIED_TOLERANCE) -> list[str]:
    """Verify split positive rates stay within tolerance of each other."""
    splits = stats.get("splits", {})
    rates = [splits[name]["positive_rate"] for name in ("train", "val", "test") if name in splits]
    if len(rates) < 2:
        return ["stats.json missing split positive rates"]
    spread = max(rates) - min(rates)
    if spread > tolerance:
        return [f"positive rate spread {spread:.6f} exceeds tolerance {tolerance}"]
    return []


def verify_dvc_pointer(root: Path, dataset: dict[str, Any]) -> list[str]:
    """Verify DVC pointer file exists for a ready dataset."""
    if not dataset.get("ready"):
        return []
    pointer = dvc_pointer_path(root, dataset)
    if not pointer.is_file():
        return [f"missing DVC pointer: {pointer}"]
    text = pointer.read_text(encoding="utf-8")
    if "outs:" not in text or "path:" not in text:
        return [f"invalid DVC pointer format: {pointer}"]
    return []


def collect_open_data_issues(
    root: Path,
    *,
    manifest_path: Path | None = None,
    dataset_id: str = "higgs_v1",
) -> list[str]:
    """Collect validation issues for an open dataset manifest entry."""
    manifest_file = manifest_path or (root / "data" / "open" / "manifest.json")
    issues: list[str] = []
    try:
        manifest = load_manifest(manifest_file)
        dataset = get_dataset(manifest, dataset_id)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return [str(exc)]

    if not dataset.get("ready"):
        return issues

    # Image-text packs (caption corpora): processed stats + pairs + optional checksums.
    if dataset.get("modality") == "image_text":
        out_dir = processed_dir(root, dataset)
        issues.extend(verify_checksums(dataset, out_dir))
        for key in ("stats", "pairs"):
            filename = dataset.get("files", {}).get(key)
            if not filename:
                issues.append(f"missing files.{key} in manifest for {dataset_id}")
                continue
            path = out_dir / filename
            if not path.is_file():
                issues.append(f"missing {key}: {path}")
        return issues

    # Image packs: stats + split indices only (no tabular parquet / DVC pointer yet).
    if dataset.get("modality") == "images":
        out_dir = processed_dir(root, dataset)
        issues.extend(verify_checksums(dataset, out_dir))
        for key in ("stats", "split_indices"):
            filename = dataset.get("files", {}).get(key)
            if not filename:
                issues.append(f"missing files.{key} in manifest for {dataset_id}")
                continue
            path = out_dir / filename
            if not path.is_file():
                issues.append(f"missing {key}: {path}")
        return issues

    issues.extend(verify_dvc_pointer(root, dataset))

    out_dir = processed_dir(root, dataset)
    issues.extend(verify_checksums(dataset, out_dir))

    stats_path = out_dir / dataset["files"].get("stats", "stats.json")
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("split_method") not in ("temporal_sha_order", "temporal_crop_year"):
            issues.extend(verify_stratified_balance(stats))
    else:
        issues.append(f"missing stats: {stats_path}")

    n_features = int(dataset["n_features"])
    for split_name in ("train", "val", "test"):
        parquet_path = out_dir / dataset["files"][split_name]
        if not parquet_path.is_file():
            issues.append(f"missing parquet: {parquet_path}")
            continue
        frame = pd.read_parquet(parquet_path)
        expected_rows = dataset["row_counts"][split_name]
        if len(frame) != expected_rows:
            issues.append(f"{split_name} row count {len(frame)} != {expected_rows}")
        issues.extend(verify_parquet_frame(frame, n_features))

    return issues


def validate_open_data(root: Path, *, dataset_id: str = "higgs_v1") -> tuple[bool, list[str]]:
    """Validate open data manifest, checksums, schema, and DVC pointer."""
    issues = collect_open_data_issues(root, dataset_id=dataset_id)
    return len(issues) == 0, issues


def validate_all_ready_open_data(root: Path) -> tuple[bool, list[str]]:
    """Validate every dataset marked ready in manifest.json."""
    manifest_path = root / "data" / "open" / "manifest.json"
    manifest = load_manifest(manifest_path)
    all_issues: list[str] = []
    for dataset in manifest.get("datasets", []):
        if not dataset.get("ready"):
            continue
        # Image packs use a different on-disk contract (raw markers + processed stats).
        if dataset.get("modality") in ("images", "image_text"):
            continue
        ok, issues = validate_open_data(root, dataset_id=dataset["id"])
        if not ok:
            all_issues.extend(f"{dataset['id']}: {issue}" for issue in issues)
    return len(all_issues) == 0, all_issues
