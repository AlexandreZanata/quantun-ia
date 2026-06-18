"""Unit tests for HIGGS open dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.open_higgs import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_FEATURES,
    build_higgs_frame,
    build_stats_payload,
    compute_split_stats,
    feature_column_names,
    labels_to_binary_int,
    split_higgs_partitions,
    subsample_stratified,
    write_parquet_splits,
)


def _synthetic_higgs(n_rows: int = 10_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, N_FEATURES)).astype(np.float32)
    labels = rng.integers(0, 2, size=n_rows).astype(np.float32)
    return features, labels


def test_feature_column_names_match_schema():
    cols = feature_column_names()
    assert len(cols) == 28
    assert cols[0] == "feature_0"
    assert cols[-1] == "feature_27"


def test_subsample_stratified_exact_count():
    x, y = _synthetic_higgs(10_000)
    sub_x, sub_y = subsample_stratified(x, y, n_samples=1_000, random_state=42)
    assert sub_x.shape == (1_000, N_FEATURES)
    assert sub_y.shape == (1_000,)


def test_split_higgs_partitions_row_counts():
    x, y = _synthetic_higgs(1_150_000, seed=1)
    x_train, y_train, x_val, y_val, x_test, y_test = split_higgs_partitions(x, y)
    assert len(y_train) == 805_000
    assert len(y_val) == 172_500
    assert len(y_test) == 172_500


def test_build_higgs_frame_schema():
    x, y = _synthetic_higgs(100)
    frame = build_higgs_frame(x, y)
    assert list(frame.columns) == FEATURE_COLUMNS + [LABEL_COLUMN]
    assert frame[LABEL_COLUMN].dtype == np.int32
    assert frame[FEATURE_COLUMNS].dtypes.eq(np.float32).all()
    assert frame.isna().sum().sum() == 0


def test_labels_to_binary_int_rejects_invalid():
    bad = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    try:
        labels_to_binary_int(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_write_parquet_splits_shapes(tmp_path: Path):
    x, y = _synthetic_higgs(1_150_000, seed=2)
    x_train, y_train, x_val, y_val, x_test, y_test = split_higgs_partitions(x, y)
    train = build_higgs_frame(x_train, y_train)
    val = build_higgs_frame(x_val, y_val)
    test = build_higgs_frame(x_test, y_test)
    paths = write_parquet_splits(tmp_path, train, val, test)

    assert paths["train"].is_file()
    train_frame = pd.read_parquet(paths["train"])
    assert train_frame.shape == (805_000, N_FEATURES + 1)
    stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
    assert stats["dataset_id"] == "higgs_v1"
    assert stats["splits"]["train"]["n_rows"] == 805_000


def test_compute_split_stats_class_balance():
    x, y = _synthetic_higgs(1_000)
    frame = build_higgs_frame(x, y)
    stats = compute_split_stats(frame)
    assert stats["n_rows"] == 1_000
    assert set(stats["class_counts"].keys()) == {"0", "1"}


def test_build_stats_payload_has_splits():
    x, y = _synthetic_higgs(500)
    frame = build_higgs_frame(x, y)
    payload = build_stats_payload(frame, frame, frame)
    assert "train" in payload["splits"]
    assert payload["n_features"] == N_FEATURES
