"""Unit tests for hard temporal drift helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.hard_drift_acyd import (
    DEFAULT_TRAIN_MAX_YEAR,
    DEFAULT_VAL_YEARS,
    ensure_hard_drift_maize_processed,
    temporal_year_split,
)

ROOT = Path(__file__).resolve().parents[2]


def test_temporal_year_split_hard_drift_defaults():
    years = np.array([2015, 2016, 2017, 2018, 2019, 2022, 2023])
    train, val, test = temporal_year_split(
        years,
        train_max_year=DEFAULT_TRAIN_MAX_YEAR,
        val_years=DEFAULT_VAL_YEARS,
        test_min_year=2022,
    )
    assert train.tolist() == [True, True, False, False, False, False, False]
    assert val.tolist() == [False, False, True, True, False, False, False]
    assert test.tolist() == [False, False, False, False, False, True, True]


@pytest.mark.skipif(
    not (ROOT / "data/open/acyd_maize_brazil/raw/crop_corn_yield.csv").is_file(),
    reason="ACYD maize raw missing",
)
def test_ensure_hard_drift_maize_small_chunks(tmp_path: Path):
    out = tmp_path / "hard_drift"
    path = ensure_hard_drift_maize_processed(
        ROOT,
        force=True,
        max_feature_chunks=1,
        out_dir=out,
    )
    assert path == out
    assert (out / "train.parquet").is_file()
    assert (out / "val.parquet").is_file()
    assert (out / "test.parquet").is_file()
