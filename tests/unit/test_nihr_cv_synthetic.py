"""Unit tests — NIHR synthetic CV dataset builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.nihr_cv_synthetic import (
    COHORT_TOTAL,
    N_FEATURES,
    RAW_FEATURE_COLUMNS,
    build_nihr_frame,
    build_nihr_processed,
    encode_nihr_features,
    impute_train_only,
    load_nihr_frame,
    split_nihr_partitions,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "open" / "nihr_cv_synthetic" / "raw" / "cvd_synthetic_dataset_v0.2.csv"


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    if not RAW_CSV.is_file():
        pytest.skip("NIHR raw CSV not downloaded — run make data-open-nihr-cv")
    return load_nihr_frame(RAW_CSV)


def test_encode_nihr_features_shape(raw_frame: pd.DataFrame):
    features, labels = encode_nihr_features(raw_frame)
    assert features.shape == (COHORT_TOTAL, N_FEATURES)
    assert labels.shape == (COHORT_TOTAL,)
    assert set(np.unique(labels)) <= {0.0, 1.0}


def test_impute_train_only_removes_nan():
    x_train = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    x_val = np.array([[2.0, np.nan]], dtype=np.float32)
    x_test = np.array([[np.nan, 5.0]], dtype=np.float32)
    train, val, test = impute_train_only(x_train, x_val, x_test)
    assert not np.isnan(train).any()
    assert not np.isnan(val).any()
    assert not np.isnan(test).any()
    assert val[0, 1] == pytest.approx(4.0)


def test_build_nihr_processed_writes_splits(tmp_path: Path, raw_frame: pd.DataFrame):
    raw_path = tmp_path / "nihr.csv"
    raw_frame.to_csv(raw_path, index=False)
    out_dir = tmp_path / "processed" / "v1"
    paths = build_nihr_processed(raw_path, out_dir)
    for split in ("train", "val", "test", "stats"):
        assert paths[split].is_file()
    train = pd.read_parquet(paths["train"])
    assert list(train.columns) == [f"feature_{i}" for i in range(N_FEATURES)] + ["label"]
    assert train.isna().sum().sum() == 0
    assert len(train) == 70_000
