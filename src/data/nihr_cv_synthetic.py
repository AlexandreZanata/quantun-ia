"""NIHR synthetic cardiovascular dataset — Zenodo ingest and parquet export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.data.open_higgs import sha256_file, update_manifest_ready

NIHR_ZENODO_CSV_URL = (
    "https://zenodo.org/api/records/12567416/files/cvd_synthetic_dataset_v0.2.csv/content"
)
NIHR_LICENSE = "CC0-1.0"
NIHR_SOURCE_URL = "https://doi.org/10.5281/zenodo.12567416"
NIHR_DATASET_ID = "nihr_cv_synthetic_v1"
N_FEATURES = 13
COHORT_TOTAL = 100_000
TRAIN_ROWS = 70_000
VAL_ROWS = 15_000
TEST_ROWS = 15_000
RANDOM_STATE = 42
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]

RAW_FEATURE_COLUMNS = [
    "gender",
    "age",
    "body_mass_index",
    "smoker",
    "systolic_blood_pressure",
    "hypertension_treated",
    "family_history_of_cardiovascular_disease",
    "atrial_fibrillation",
    "chronic_kidney_disease",
    "rheumatoid_arthritis",
    "diabetes",
    "chronic_obstructive_pulmonary_disorder",
    "forced_expiratory_volume_1",
]

FEATURE_SEMANTICS = [
    "sex_male",
    "age_years",
    "body_mass_index",
    "smoker",
    "systolic_blood_pressure",
    "hypertension_treated",
    "family_history_cvd",
    "atrial_fibrillation",
    "chronic_kidney_disease",
    "rheumatoid_arthritis",
    "diabetes",
    "copd",
    "forced_expiratory_volume_1",
]


def feature_column_names(n_features: int = N_FEATURES) -> list[str]:
    """Return canonical feature column names for tabular_binary_v1."""
    return [f"feature_{i}" for i in range(n_features)]


def download_nihr_raw(raw_path: Path, *, url: str = NIHR_ZENODO_CSV_URL) -> Path:
    """Download NIHR CVD synthetic CSV from Zenodo if missing."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_file():
        urlretrieve(url, raw_path)  # noqa: S310 — pinned Zenodo URL from manifest
    return raw_path


def load_nihr_frame(raw_path: Path) -> pd.DataFrame:
    """Load raw NIHR CSV with label column."""
    frame = pd.read_csv(raw_path)
    required = set(RAW_FEATURE_COLUMNS) | {"heart_attack_or_stroke_occurred"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"NIHR CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)
    return frame


def encode_nihr_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode raw NIHR columns to float32 feature matrix and binary labels."""
    labels = frame["heart_attack_or_stroke_occurred"].to_numpy(dtype=np.float32)
    gender = (frame["gender"].astype(str).str.upper() == "M").astype(np.float32)
    numeric = frame[RAW_FEATURE_COLUMNS[1:]].astype(np.float32)
    features = np.column_stack([gender.to_numpy(dtype=np.float32), numeric.to_numpy(dtype=np.float32)])
    return features.astype(np.float32), labels


def impute_train_only(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute missing values using train statistics only."""
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train).astype(np.float32)
    x_val_imp = imputer.transform(x_val).astype(np.float32)
    x_test_imp = imputer.transform(x_test).astype(np.float32)
    return x_train_imp, x_val_imp, x_test_imp


def split_nihr_partitions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_rows: int = VAL_ROWS,
    test_rows: int = TEST_ROWS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / val / test split."""
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


def build_nihr_frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
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
    """Aggregate stats.json for NIHR processed v1."""
    return {
        "dataset_id": NIHR_DATASET_ID,
        "license": NIHR_LICENSE,
        "source_url": NIHR_SOURCE_URL,
        "source_mode": "zenodo_csv",
        "feature_semantics": FEATURE_SEMANTICS,
        "n_features": N_FEATURES,
        "random_state": RANDOM_STATE,
        "cohort_total": COHORT_TOTAL,
        "imputation": "train_median",
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


def build_nihr_processed(
    raw_path: Path,
    out_dir: Path,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, Path]:
    """End-to-end build from Zenodo CSV to imputed parquet splits."""
    frame = load_nihr_frame(raw_path)
    if len(frame) != COHORT_TOTAL:
        msg = f"expected {COHORT_TOTAL} rows, got {len(frame)}"
        raise ValueError(msg)
    features, labels = encode_nihr_features(frame)
    x_train, y_train, x_val, y_val, x_test, y_test = split_nihr_partitions(
        features,
        labels,
        random_state=random_state,
    )
    x_train, x_val, x_test = impute_train_only(x_train, x_val, x_test)
    train = build_nihr_frame(x_train, y_train)
    val = build_nihr_frame(x_val, y_val)
    test = build_nihr_frame(x_test, y_test)
    return write_parquet_splits(out_dir, train, val, test)


def update_nihr_manifest_ready(manifest_path: Path, processed_dir: Path) -> dict[str, Any]:
    """Mark nihr_cv_synthetic_v1 ready with checksums."""
    return update_manifest_ready(
        manifest_path,
        processed_dir,
        dataset_id=NIHR_DATASET_ID,
    )
