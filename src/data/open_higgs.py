"""HIGGS open dataset builder — download, subsample, and export parquet splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HIGGS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
)
HIGGS_LICENSE = "CC0-1.0"
N_FEATURES = 28
SUBSAMPLE_TOTAL = 1_150_000
TRAIN_ROWS = 805_000
VAL_ROWS = 172_500
TEST_ROWS = 172_500
RANDOM_STATE = 42
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]
LABEL_COLUMN = "label"


def feature_column_names(n_features: int = N_FEATURES) -> list[str]:
    """Return canonical feature column names for tabular_binary_v1."""
    return [f"feature_{i}" for i in range(n_features)]


def download_higgs_raw(raw_path: Path, *, url: str = HIGGS_URL) -> Path:
    """Download HIGGS.csv.gz if missing."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_file():
        urlretrieve(url, raw_path)  # noqa: S310 — pinned UCI URL from manifest
    return raw_path


def load_higgs_array(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HIGGS CSV into feature matrix X and label vector y."""
    frame = pd.read_csv(
        raw_path,
        header=None,
        dtype=np.float32,
        compression="gzip" if raw_path.suffix == ".gz" else None,
    )
    if frame.shape[1] != N_FEATURES + 1:
        msg = f"expected {N_FEATURES + 1} columns, got {frame.shape[1]}"
        raise ValueError(msg)
    labels = frame.iloc[:, 0].to_numpy(dtype=np.float32)
    features = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    return features, labels


def subsample_stratified(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int = SUBSAMPLE_TOTAL,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to exactly n_samples rows."""
    if n_samples > len(labels):
        msg = f"cannot subsample {n_samples} rows from {len(labels)}"
        raise ValueError(msg)
    indices = np.arange(len(labels))
    selected, _ = train_test_split(
        indices,
        train_size=n_samples,
        stratify=labels,
        random_state=random_state,
    )
    return features[selected], labels[selected]


def split_higgs_partitions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_rows: int = VAL_ROWS,
    test_rows: int = TEST_ROWS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split subsample into stratified train / val / test partitions."""
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_rows,
        stratify=labels,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rows,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )
    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
        features[test_idx],
        labels[test_idx],
    )


def labels_to_binary_int(labels: np.ndarray) -> np.ndarray:
    """Convert HIGGS labels to int32 {0, 1}."""
    unique = set(np.unique(labels).tolist())
    if not unique <= {0.0, 1.0}:
        msg = f"unexpected label values: {sorted(unique)}"
        raise ValueError(msg)
    return labels.astype(np.int32)


def build_higgs_frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Build export frame matching tabular_binary_v1 schema."""
    frame = pd.DataFrame(features.astype(np.float32), columns=FEATURE_COLUMNS)
    frame[LABEL_COLUMN] = labels_to_binary_int(labels)
    return frame


def compute_split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute class balance and feature means for a split."""
    label_counts = frame[LABEL_COLUMN].value_counts().sort_index()
    pos = int(label_counts.get(1, 0))
    neg = int(label_counts.get(0, 0))
    total = len(frame)
    feature_means = {
        col: float(frame[col].mean()) for col in FEATURE_COLUMNS
    }
    return {
        "n_rows": total,
        "class_counts": {"0": neg, "1": pos},
        "positive_rate": round(pos / total, 6) if total else 0.0,
        "feature_means": feature_means,
    }


def build_stats_payload(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source_url: str = HIGGS_URL,
) -> dict[str, Any]:
    """Aggregate stats.json content for processed HIGGS v1."""
    return {
        "dataset_id": "higgs_v1",
        "license": HIGGS_LICENSE,
        "source_url": source_url,
        "n_features": N_FEATURES,
        "random_state": RANDOM_STATE,
        "subsample_total": SUBSAMPLE_TOTAL,
        "splits": {
            "train": compute_split_stats(train),
            "val": compute_split_stats(val),
            "test": compute_split_stats(test),
        },
    }


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet_splits(
    out_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Path]:
    """Write train/val/test parquet files and stats.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)
    stats_path = out_dir / "stats.json"
    stats_path.write_text(
        json.dumps(build_stats_payload(train, val, test), indent=2) + "\n",
        encoding="utf-8",
    )
    paths["stats"] = stats_path
    return paths


def build_higgs_processed(
    raw_path: Path,
    out_dir: Path,
    *,
    subsample_total: int = SUBSAMPLE_TOTAL,
    random_state: int = RANDOM_STATE,
) -> dict[str, Path]:
    """End-to-end build from raw gzip CSV to parquet splits."""
    features, labels = load_higgs_array(raw_path)
    sub_x, sub_y = subsample_stratified(
        features,
        labels,
        n_samples=subsample_total,
        random_state=random_state,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = split_higgs_partitions(
        sub_x,
        sub_y,
        random_state=random_state,
    )
    train = build_higgs_frame(x_train, y_train)
    val = build_higgs_frame(x_val, y_val)
    test = build_higgs_frame(x_test, y_test)
    return write_parquet_splits(out_dir, train, val, test)


def update_manifest_ready(
    manifest_path: Path,
    processed_dir: Path,
    *,
    dataset_id: str = "higgs_v1",
) -> dict[str, Any]:
    """Mark dataset ready and attach SHA-256 checksums in manifest.json."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = next(d for d in manifest["datasets"] if d["id"] == dataset_id)
    files = dataset["files"]
    checksums = {
        key: sha256_file(processed_dir / filename)
        for key, filename in files.items()
    }
    dataset["checksums"] = checksums
    dataset["ready"] = True
    manifest["updated_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
