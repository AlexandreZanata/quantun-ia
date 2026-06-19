"""GoBug file-level defect dataset — go-bug-collector ingest and parquet export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.data.open_higgs import update_manifest_ready

GOBUG_BUG_URL = (
    "https://raw.githubusercontent.com/ecylmz/go-bug-collector/main/"
    "file_data/combined/file_bug_metrics.csv"
)
GOBUG_NON_BUG_URL = (
    "https://raw.githubusercontent.com/ecylmz/go-bug-collector/main/"
    "file_data/combined/file_non_bug_metrics.csv"
)
GOBUG_LICENSE = "See IEEE DataPort GoBug + go-bug-collector LICENSE-data"
GOBUG_SOURCE_URL = "https://doi.org/10.21227/bk5q-fs89"
GOBUG_DATASET_ID = "code_defects_gobug_v1"
N_FEATURES = 23
RANDOM_STATE = 42
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]
ID_COLUMNS = ("project", "file_path", "sha")
METRIC_COLUMNS = [
    "nloc",
    "complexity",
    "token_count",
    "method_count",
    "commit_count",
    "authors_count",
    "avg_method_param_count",
    "import_count",
    "cyclo_per_loc",
    "comment_ratio",
    "struct_count",
    "interface_count",
    "loop_count",
    "error_handling_count",
    "goroutine_count",
    "channel_count",
    "defer_count",
    "context_usage_count",
    "json_tag_count",
    "variadic_function_count",
    "pointer_receiver_count",
    "avg_method_complexity",
    "avg_methods_token_count",
]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def feature_column_names(n_features: int = N_FEATURES) -> list[str]:
    """Return canonical feature column names for tabular_binary_v1."""
    return [f"feature_{i}" for i in range(n_features)]


def download_gobug_raw(bug_path: Path, non_bug_path: Path) -> tuple[Path, Path]:
    """Download GoBug combined CSVs from go-bug-collector if missing."""
    bug_path.parent.mkdir(parents=True, exist_ok=True)
    if not bug_path.is_file():
        urlretrieve(GOBUG_BUG_URL, bug_path)  # noqa: S310
    if not non_bug_path.is_file():
        urlretrieve(GOBUG_NON_BUG_URL, non_bug_path)  # noqa: S310
    return bug_path, non_bug_path


def load_gobug_frame(bug_path: Path, non_bug_path: Path) -> pd.DataFrame:
    """Load and merge buggy / non-buggy file-level rows with label column."""
    bug = pd.read_csv(bug_path)
    non_bug = pd.read_csv(non_bug_path)
    for frame, label in ((bug, 1), (non_bug, 0)):
        missing = set(METRIC_COLUMNS + list(ID_COLUMNS)) - set(frame.columns)
        if missing:
            msg = f"GoBug CSV missing columns: {sorted(missing)}"
            raise ValueError(msg)
    bug[LABEL_COLUMN] = 1
    non_bug[LABEL_COLUMN] = 0
    return pd.concat([bug, non_bug], ignore_index=True)


def temporal_split_by_sha(
    frame: pd.DataFrame,
    *,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal proxy split: sort by commit sha, allocate contiguous time blocks."""
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        msg = "split fractions must sum to 1"
        raise ValueError(msg)
    ordered = frame.sort_values(["sha", "project", "file_path"]).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def build_gobug_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract float32 feature matrix and binary labels."""
    features = frame[METRIC_COLUMNS].astype(np.float32).to_numpy()
    labels = frame[LABEL_COLUMN].astype(np.float32).to_numpy()
    if np.isnan(features).any():
        msg = "NaN values forbidden in GoBug metrics after export"
        raise ValueError(msg)
    return features, labels


def build_gobug_frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Build tabular_binary_v1 dataframe."""
    data: dict[str, Any] = {col: features[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    data[LABEL_COLUMN] = labels.astype(np.int64)
    return pd.DataFrame(data)


def compute_split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute class balance for a split."""
    label_counts = frame[LABEL_COLUMN].value_counts().sort_index()
    pos = int(label_counts.get(1, 0))
    neg = int(label_counts.get(0, 0))
    total = len(frame)
    return {
        "n_rows": total,
        "class_counts": {"0": neg, "1": pos},
        "positive_rate": round(pos / total, 6) if total else 0.0,
    }


def build_stats_payload(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate stats.json for GoBug processed v1."""
    return {
        "dataset_id": GOBUG_DATASET_ID,
        "license": GOBUG_LICENSE,
        "source_url": GOBUG_SOURCE_URL,
        "source_mode": "go_bug_collector_combined",
        "feature_semantics": METRIC_COLUMNS,
        "n_features": N_FEATURES,
        "split_method": "temporal_sha_order",
        "split_fractions": {
            "train": TRAIN_FRAC,
            "val": VAL_FRAC,
            "test": TEST_FRAC,
        },
        "splits": {
            "train": compute_split_stats(train),
            "val": compute_split_stats(val),
            "test": compute_split_stats(test),
        },
    }


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


def build_gobug_processed(
    bug_path: Path,
    non_bug_path: Path,
    out_dir: Path,
) -> dict[str, Path]:
    """End-to-end build from go-bug-collector CSVs to temporal parquet splits."""
    raw = load_gobug_frame(bug_path, non_bug_path)
    train_raw, val_raw, test_raw = temporal_split_by_sha(raw)
    x_train, y_train = build_gobug_features(train_raw)
    x_val, y_val = build_gobug_features(val_raw)
    x_test, y_test = build_gobug_features(test_raw)
    return write_parquet_splits(
        out_dir,
        build_gobug_frame(x_train, y_train),
        build_gobug_frame(x_val, y_val),
        build_gobug_frame(x_test, y_test),
    )


def update_gobug_manifest_ready(manifest_path: Path, processed_dir: Path) -> dict[str, Any]:
    """Mark code_defects_gobug_v1 ready with checksums and row counts."""
    stats = json.loads((processed_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = next(d for d in manifest["datasets"] if d["id"] == GOBUG_DATASET_ID)
    splits = stats["splits"]
    dataset["row_counts"] = {
        "total": sum(splits[name]["n_rows"] for name in ("train", "val", "test")),
        "train": splits["train"]["n_rows"],
        "val": splits["val"]["n_rows"],
        "test": splits["test"]["n_rows"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return update_manifest_ready(manifest_path, processed_dir, dataset_id=GOBUG_DATASET_ID)
