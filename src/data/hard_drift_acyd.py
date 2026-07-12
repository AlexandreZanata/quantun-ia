"""Hard temporal drift splits for ACYD maize (Phase C / exp_094).

Standard processed parquet has no ``year`` column. This module rebuilds
tabular splits from raw ACYD maize with a stricter temporal protocol:

- train: year ≤ 2016
- val: year ∈ {2017, 2018}
- test: year ≥ 2022
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.open_acyd import build_acyd_maize_processed, temporal_year_split

HARD_DRIFT_DATASET_ID = "acyd_maize_brazil_hard_drift_v1"
HARD_DRIFT_REL_PATH = Path("data/open/acyd_maize_brazil/processed/hard_drift_v1")
DEFAULT_TRAIN_MAX_YEAR = 2016
DEFAULT_VAL_YEARS = (2017, 2018)
DEFAULT_TEST_MIN_YEAR = 2022


@dataclass(frozen=True)
class HardDriftMaizeSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    n_train: int
    n_val: int
    n_test: int
    processed_dir: Path


def hard_drift_processed_dir(root: Path) -> Path:
    return root / HARD_DRIFT_REL_PATH


def hard_drift_ready(root: Path) -> bool:
    out = hard_drift_processed_dir(root)
    return all((out / name).is_file() for name in ("train.parquet", "val.parquet", "test.parquet"))


def ensure_hard_drift_maize_processed(
    root: Path,
    *,
    force: bool = False,
    max_feature_chunks: int | None = None,
    train_max_year: int = DEFAULT_TRAIN_MAX_YEAR,
    val_years: tuple[int, ...] = DEFAULT_VAL_YEARS,
    test_min_year: int = DEFAULT_TEST_MIN_YEAR,
    out_dir: Path | None = None,
) -> Path:
    """Build hard-drift parquet under processed/hard_drift_v1 if missing."""
    target = out_dir or hard_drift_processed_dir(root)
    ready = all(
        (target / name).is_file() for name in ("train.parquet", "val.parquet", "test.parquet")
    )
    if ready and not force:
        return target
    raw_dir = root / "data" / "open" / "acyd_maize_brazil" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    build_acyd_maize_processed(
        raw_dir,
        target,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
        max_feature_chunks=max_feature_chunks,
    )
    return target


def _cap_rows(
    x: np.ndarray,
    y: np.ndarray,
    n_rows: int | None,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows is None or n_rows <= 0 or n_rows >= len(y):
        return x, y
    idx = np.arange(len(y))
    selected, _ = train_test_split(
        idx,
        train_size=int(n_rows),
        stratify=y,
        random_state=random_state,
    )
    return x[selected], y[selected]


def load_hard_drift_maize_splits(
    root: Path,
    *,
    n_train_rows: int | None = None,
    n_val_rows: int | None = None,
    random_state: int = 42,
    ensure: bool = True,
    max_feature_chunks: int | None = None,
    processed_dir: Path | None = None,
) -> HardDriftMaizeSplits:
    """Load hard-drift maize arrays with train-only StandardScaler."""
    import pandas as pd

    target = processed_dir or hard_drift_processed_dir(root)
    if ensure:
        ensure_hard_drift_maize_processed(
            root,
            max_feature_chunks=max_feature_chunks,
            out_dir=target,
        )
    ready = all(
        (target / name).is_file() for name in ("train.parquet", "val.parquet", "test.parquet")
    )
    if not ready:
        msg = f"hard-drift maize missing under {target}"
        raise FileNotFoundError(msg)

    feature_cols = [f"feature_{i}" for i in range(37)]

    def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
        frame = pd.read_parquet(target / f"{name}.parquet")
        return (
            frame[feature_cols].to_numpy(dtype=np.float32),
            frame["label"].to_numpy(dtype=np.float32),
        )

    x_train, y_train = _load("train")
    x_val, y_val = _load("val")
    x_test, y_test = _load("test")
    x_train, y_train = _cap_rows(x_train, y_train, n_train_rows, random_state=random_state)
    x_val, y_val = _cap_rows(x_val, y_val, n_val_rows, random_state=random_state + 1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return HardDriftMaizeSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        processed_dir=target,
    )


__all__ = [
    "DEFAULT_TEST_MIN_YEAR",
    "DEFAULT_TRAIN_MAX_YEAR",
    "DEFAULT_VAL_YEARS",
    "HARD_DRIFT_DATASET_ID",
    "HardDriftMaizeSplits",
    "ensure_hard_drift_maize_processed",
    "hard_drift_processed_dir",
    "hard_drift_ready",
    "load_hard_drift_maize_splits",
    "temporal_year_split",
]
