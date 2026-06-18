"""Unit tests for Synthea cardiovascular risk cohort builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.synthea_cv_risk import (
    N_FEATURES,
    build_cohort_frame,
    build_synthea_processed,
    extract_patient_row,
    feature_column_names,
    simulate_clinical_cohort,
    split_cohort_partitions,
    subsample_to_target,
)


def test_feature_column_names_count():
    assert len(feature_column_names()) == 40


def test_simulate_clinical_cohort_shape():
    features, labels = simulate_clinical_cohort(n_rows=1000, random_state=42)
    assert features.shape == (1000, N_FEATURES)
    assert labels.shape == (1000,)
    assert set(np.unique(labels)) <= {0.0, 1.0}


def test_split_cohort_partitions_row_counts():
    features, labels = simulate_clinical_cohort(n_rows=1_000_000, random_state=1)
    x_train, y_train, x_val, y_val, x_test, y_test = split_cohort_partitions(features, labels)
    assert len(y_train) == 700_000
    assert len(y_val) == 150_000
    assert len(y_test) == 150_000


def test_build_cohort_frame_schema():
    features, labels = simulate_clinical_cohort(n_rows=50, random_state=0)
    frame = build_cohort_frame(features, labels)
    assert len(frame.columns) == N_FEATURES + 1
    assert frame["label"].dtype == np.int32
    assert frame.isna().sum().sum() == 0


def test_subsample_to_target_exact():
    features, labels = simulate_clinical_cohort(n_rows=5000, random_state=3)
    sub_x, sub_y = subsample_to_target(features, labels, n_samples=1000, random_state=42)
    assert sub_x.shape == (1000, N_FEATURES)


def test_extract_patient_row_from_minimal_bundle():
    bundle = {
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "gender": "male",
                    "birthDate": "1960-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"coding": [{"code": "22298006"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"coding": [{"code": "8480-6"}]},
                    "valueQuantity": {"value": 140},
                }
            },
        ]
    }
    row, label = extract_patient_row(bundle)
    assert row.shape == (N_FEATURES,)
    assert label == 1.0


def test_build_synthea_processed_writes_parquet(tmp_path: Path):
    out_dir = tmp_path / "v1"
    paths, source_mode = build_synthea_processed(out_dir)
    assert source_mode == "clinical_simulation"
    assert paths["train"].is_file()
    train = pd.read_parquet(paths["train"])
    assert train.shape == (700_000, N_FEATURES + 1)
    stats = json.loads(paths["stats"].read_text(encoding="utf-8"))
    assert stats["dataset_id"] == "synthea_cv_risk_v1"
