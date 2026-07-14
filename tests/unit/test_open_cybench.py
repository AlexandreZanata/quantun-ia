"""Unit tests for CY-Bench maize sample builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.open_cybench import (
    CYBENCH_DATASET_ID,
    binarize_low_yield,
    build_cybench_maize_processed,
    ensure_manifest_entry,
    load_cybench_us_feature_table,
    split_cybench_years,
    sync_cybench_sample_us,
    update_cybench_manifest_ready,
)


def _write_sample_csvs(sample_dir: Path) -> Path:
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year, yield_v, adm, f1 in [
        (2010, 10.0, "A", 1.0),
        (2011, 20.0, "A", 2.0),
        (2012, 15.0, "B", 3.0),
        (2013, 25.0, "B", 4.0),
        (2014, 14.0, "C", 4.5),
        (2015, 22.0, "C", 5.0),
        (2016, 12.0, "D", 5.5),
        (2017, 18.0, "D", 6.0),
    ]:
        rows.append(
            {
                "adm_id": adm,
                "year": year,
                "f1": f1,
                "yield": yield_v,
                "yield_lag_1": yield_v - 1.0,
            }
        )
    frame = pd.DataFrame(rows)
    # Official AgML sample is pre-split; keep identical schema.
    train = frame[frame["year"] <= 2015]
    test = frame[frame["year"] >= 2016]
    train.to_csv(sample_dir / "grain_maize_US_train.csv", index=False)
    test.to_csv(sample_dir / "grain_maize_US_test.csv", index=False)
    return sample_dir


def test_binarize_and_split_roundtrip():
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


def test_binarize_requires_train_years():
    frame = pd.DataFrame({"year": [2020, 2021], "yield": [1.0, 2.0]})
    with pytest.raises(ValueError, match="no rows"):
        binarize_low_yield(frame, train_max_year=2011)


def test_split_rejects_empty_partition():
    frame = pd.DataFrame(
        {
            "year": [2010, 2011],
            "yield": [1.0, 2.0],
            "label": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="empty"):
        split_cybench_years(frame)


def test_load_and_build_processed(tmp_path: Path):
    sample = _write_sample_csvs(tmp_path / "sample_us")
    loaded = load_cybench_us_feature_table(sample)
    assert len(loaded) == 8
    out = tmp_path / "processed"
    stats = build_cybench_maize_processed(sample, out)
    assert stats["dataset_id"] == CYBENCH_DATASET_ID
    assert "yield_lag_1" not in stats["feature_names"]
    assert "label" not in stats["feature_names"]
    assert stats["n_features"] == 1
    assert (out / "train.parquet").is_file()
    assert (out / "stats.json").is_file()
    assert set(stats["checksums"]) >= {"train", "val", "test", "stats"}


def test_sync_prefers_existing_csvs(tmp_path: Path):
    raw = tmp_path / "raw"
    sample = _write_sample_csvs(raw / "sample_us")
    dest = sync_cybench_sample_us(raw)
    assert dest == sample


def test_sync_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing"):
        sync_cybench_sample_us(tmp_path / "raw")


def test_sync_from_sample_repo(tmp_path: Path):
    repo = tmp_path / "sample_data"
    src = repo / "features" / "maize" / "US"
    src.mkdir(parents=True)
    (src / "grain_maize_US_train.csv").write_text("adm_id,year,f1,yield\nA,2010,1,10\n")
    (src / "grain_maize_US_test.csv").write_text("adm_id,year,f1,yield\nB,2016,2,12\n")
    (repo / "README.md").write_text("sample\n")
    dest = sync_cybench_sample_us(tmp_path / "raw", sample_repo=repo)
    assert (dest / "grain_maize_US_train.csv").is_file()
    assert (tmp_path / "raw" / "SOURCE.md").is_file()


def test_manifest_ensure_and_ready(tmp_path: Path):
    sample = _write_sample_csvs(tmp_path / "sample_us")
    processed = tmp_path / "processed"
    build_cybench_maize_processed(sample, processed)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"datasets": [], "updated_at": "2020-01-01"}) + "\n")
    ensure_manifest_entry(manifest_path)
    ensure_manifest_entry(manifest_path)  # idempotent
    data = json.loads(manifest_path.read_text())
    assert sum(1 for d in data["datasets"] if d["id"] == CYBENCH_DATASET_ID) == 1
    update_cybench_manifest_ready(manifest_path, processed)
    ready = json.loads(manifest_path.read_text())
    entry = next(d for d in ready["datasets"] if d["id"] == CYBENCH_DATASET_ID)
    assert entry["n_features"] == 1
    assert entry["row_counts"]["total"] == 8


def test_build_excludes_label_and_yield_lags_real_sample(tmp_path: Path):
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


def test_load_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_cybench_us_feature_table(tmp_path)
