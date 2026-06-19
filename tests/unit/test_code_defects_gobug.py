"""Unit tests — GoBug file-level defect dataset builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.code_defects_gobug import (
    METRIC_COLUMNS,
    N_FEATURES,
    build_gobug_processed,
    load_gobug_frame,
    temporal_split_by_sha,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "open" / "code_defects_gobug" / "raw" / "combined"


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    bug = RAW_DIR / "file_bug_metrics.csv"
    non_bug = RAW_DIR / "file_non_bug_metrics.csv"
    if not bug.is_file() or not non_bug.is_file():
        pytest.skip("GoBug raw CSVs missing — run make data-open-gobug")
    return load_gobug_frame(bug, non_bug)


def test_load_gobug_frame_labels(raw_frame: pd.DataFrame):
    assert "label" in raw_frame.columns
    assert set(raw_frame["label"].unique()) == {0, 1}
    assert len(raw_frame) > 30_000


def test_temporal_split_fractions(raw_frame: pd.DataFrame):
    train, val, test = temporal_split_by_sha(raw_frame)
    total = len(raw_frame)
    assert len(train) + len(val) + len(test) == total
    assert len(train) == pytest.approx(total * 0.70, abs=2)
    assert len(val) == pytest.approx(total * 0.15, abs=2)


def test_build_gobug_processed_writes_splits(tmp_path: Path, raw_frame: pd.DataFrame):
    bug = tmp_path / "bug.csv"
    non_bug = tmp_path / "non_bug.csv"
    raw_frame[raw_frame["label"] == 1].drop(columns=["label"]).to_csv(bug, index=False)
    raw_frame[raw_frame["label"] == 0].drop(columns=["label"]).to_csv(non_bug, index=False)
    out_dir = tmp_path / "processed" / "v1"
    paths = build_gobug_processed(bug, non_bug, out_dir)
    train = pd.read_parquet(paths["train"])
    assert list(train.columns) == [f"feature_{i}" for i in range(N_FEATURES)] + ["label"]
    assert train.isna().sum().sum() == 0
    assert len(METRIC_COLUMNS) == N_FEATURES
