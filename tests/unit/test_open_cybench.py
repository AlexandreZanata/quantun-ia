"""Unit tests for CY-Bench maize sample builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.open_cybench import (
    binarize_low_yield,
    build_cybench_maize_processed,
    split_cybench_years,
)


def test_binarize_and_split_roundtrip(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "adm_id": ["A", "A", "B", "B", "C", "C"],
            "year": [2010, 2011, 2012, 2013, 2016, 2017],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "yield": [10.0, 20.0, 15.0, 25.0, 12.0, 18.0],
        }
    )
    labeled, thr = binarize_low_yield(frame, train_max_year=2011)
    assert thr == 15.0
    splits = split_cybench_years(labeled)
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2


def test_build_excludes_label_and_yield_lags(tmp_path: Path):
    sample = Path("data/open/cybench_maize/raw/sample_us")
    if not (sample / "grain_maize_US_train.csv").is_file():
        return
    out = tmp_path / "processed"
    stats = build_cybench_maize_processed(sample, out)
    names = stats["feature_names"]
    assert "label" not in names
    assert not any(n.startswith("yield") for n in names)
    assert stats["n_features"] == 26
    train = pd.read_parquet(out / "train.parquet")
    assert "label" in train.columns
    assert all(c.startswith("feature_") or c == "label" for c in train.columns)
