"""Continual crop-year splits for ACYD maize (Phase D / exp_098).

Standard processed parquet has no ``year`` column. This module rebuilds
tabular splits from raw ACYD maize and keeps ``year`` for sequential fine-tune.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.open_acyd import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    N_FEATURES,
    build_binary_labels_below_state_median,
    build_export_frame,
    crop_spec,
    extract_acyd_feature_matrix,
    join_yield_features,
    load_crop_yield,
    load_feature_chunks,
    temporal_year_split,
)

CONTINUAL_REL_PATH = Path("data/open/acyd_maize_brazil/processed/continual_v1")
DEFAULT_TRAIN_MAX_YEAR = 2018
DEFAULT_VAL_YEARS = (2019, 2020, 2021)
DEFAULT_TEST_MIN_YEAR = 2022


@dataclass(frozen=True)
class ContinualCropYearSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    years_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    years_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    train_years: tuple[int, ...]
    n_train: int
    n_val: int
    n_test: int
    processed_dir: Path


def continual_processed_dir(root: Path) -> Path:
    return root / CONTINUAL_REL_PATH


def continual_ready(root: Path) -> bool:
    out = continual_processed_dir(root)
    return all(
        (out / name).is_file()
        for name in ("train.parquet", "val.parquet", "test.parquet", "stats.json")
    )


def _build_frames_with_year(
    raw_dir: Path,
    *,
    train_max_year: int,
    val_years: tuple[int, ...],
    test_min_year: int,
    max_feature_chunks: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = crop_spec("maize")
    yield_path = raw_dir / spec["yield_file"]
    feature_dir = raw_dir / "features"
    if not yield_path.is_file():
        msg = f"yield file missing: {yield_path}"
        raise FileNotFoundError(msg)

    chunk_paths = sorted(feature_dir.glob("features_chunk_*.csv"))
    if max_feature_chunks is not None:
        chunk_paths = chunk_paths[: int(max_feature_chunks)]
    if not chunk_paths:
        msg = f"no feature chunks under {feature_dir}"
        raise FileNotFoundError(msg)

    yield_frame = load_crop_yield(yield_path, crop="maize")
    feature_frame = load_feature_chunks(chunk_paths)
    merged = join_yield_features(yield_frame, feature_frame)
    labels = build_binary_labels_below_state_median(
        merged, yield_column=spec["yield_column"]
    )
    features = extract_acyd_feature_matrix(merged)
    export = build_export_frame(features, labels)
    valid_mask = ~export.replace([np.inf, -np.inf], np.nan).isna().any(axis=1).to_numpy()
    export = export.loc[valid_mask].reset_index(drop=True)
    years = merged.loc[valid_mask, "year"].to_numpy(dtype=int)
    export["year"] = years

    train_mask, val_mask, test_mask = temporal_year_split(
        years,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
    )
    train = export.loc[train_mask].reset_index(drop=True)
    val = export.loc[val_mask].reset_index(drop=True)
    test = export.loc[test_mask].reset_index(drop=True)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        msg = f"empty temporal split: train={len(train)} val={len(val)} test={len(test)}"
        raise ValueError(msg)

    def _split_stats(frame: pd.DataFrame) -> dict[str, Any]:
        pos = int((frame[LABEL_COLUMN] == 1).sum())
        neg = int((frame[LABEL_COLUMN] == 0).sum())
        total = len(frame)
        return {
            "n_rows": total,
            "class_counts": {"0": neg, "1": pos},
            "positive_rate": round(pos / total, 6) if total else 0.0,
            "year_min": int(frame["year"].min()),
            "year_max": int(frame["year"].max()),
            "n_years": int(frame["year"].nunique()),
        }

    stats = {
        "dataset_id": "acyd_maize_brazil_continual_v1",
        "n_features": N_FEATURES,
        "split_method": "temporal_crop_year_with_year_column",
        "temporal_split": {
            "train_max_year": train_max_year,
            "val_years": list(val_years),
            "test_min_year": test_min_year,
        },
        "splits": {
            "train": _split_stats(train),
            "val": _split_stats(val),
            "test": _split_stats(test),
        },
        "train_years": sorted(int(y) for y in train["year"].unique()),
    }
    return train, val, test, stats


def ensure_continual_maize_processed(
    root: Path,
    *,
    force: bool = False,
    max_feature_chunks: int | None = None,
    train_max_year: int = DEFAULT_TRAIN_MAX_YEAR,
    val_years: tuple[int, ...] = DEFAULT_VAL_YEARS,
    test_min_year: int = DEFAULT_TEST_MIN_YEAR,
    out_dir: Path | None = None,
) -> Path:
    """Build continual parquet (with year) under processed/continual_v1 if missing."""
    target = out_dir or continual_processed_dir(root)
    ready = all(
        (target / name).is_file()
        for name in ("train.parquet", "val.parquet", "test.parquet", "stats.json")
    )
    if ready and not force:
        return target

    raw_dir = root / "data" / "open" / "acyd_maize_brazil" / "raw"
    train, val, test, stats = _build_frames_with_year(
        raw_dir,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
        max_feature_chunks=max_feature_chunks,
    )
    target.mkdir(parents=True, exist_ok=True)
    train.to_parquet(target / "train.parquet", index=False)
    val.to_parquet(target / "val.parquet", index=False)
    test.to_parquet(target / "test.parquet", index=False)
    (target / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return target


def _cap_rows_with_years(
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    n_rows: int | None,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_rows is None or n_rows <= 0 or n_rows >= len(y):
        return x, y, years
    idx = np.arange(len(y))
    selected, _ = train_test_split(
        idx,
        train_size=int(n_rows),
        stratify=y,
        random_state=random_state,
    )
    return x[selected], y[selected], years[selected]


def load_continual_crop_year_splits(
    root: Path,
    *,
    n_train_rows: int | None = None,
    n_val_rows: int | None = None,
    random_state: int = 42,
    ensure: bool = True,
    max_feature_chunks: int | None = None,
    processed_dir: Path | None = None,
) -> ContinualCropYearSplits:
    """Load continual maize arrays with train-only StandardScaler + year vectors."""
    target = processed_dir or continual_processed_dir(root)
    if ensure:
        ensure_continual_maize_processed(
            root,
            max_feature_chunks=max_feature_chunks,
            out_dir=target,
        )

    def _load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = pd.read_parquet(target / f"{name}.parquet")
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = frame[LABEL_COLUMN].to_numpy(dtype=np.float32)
        years = frame["year"].to_numpy(dtype=np.int32)
        return x, y, years

    x_train, y_train, years_train = _load("train")
    x_val, y_val, years_val = _load("val")
    x_test, y_test, _years_test = _load("test")

    x_train, y_train, years_train = _cap_rows_with_years(
        x_train, y_train, years_train, n_train_rows, random_state=random_state
    )
    x_val, y_val, years_val = _cap_rows_with_years(
        x_val, y_val, years_val, n_val_rows, random_state=random_state + 1
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_years = tuple(int(y) for y in sorted(np.unique(years_train).tolist()))
    return ContinualCropYearSplits(
        x_train=x_train,
        y_train=y_train,
        years_train=years_train,
        x_val=x_val,
        y_val=y_val,
        years_val=years_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        train_years=train_years,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        processed_dir=target,
    )


def rows_for_year(
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    year: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = years == int(year)
    if not np.any(mask):
        msg = f"no rows for year={year}"
        raise ValueError(msg)
    return x[mask], y[mask]
