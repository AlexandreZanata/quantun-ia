"""ACYD Brazil open dataset builder — soybean yield + climate tabular features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.data.open_higgs import sha256_file, write_parquet_splits

ACYD_HF_REPO = "notadib/ACYD"
ACYD_HF_BASE = f"https://huggingface.co/datasets/{ACYD_HF_REPO}/resolve/main"
ACYD_SOURCE_URL = "https://huggingface.co/datasets/notadib/ACYD"
ACYD_LICENSE = "CC-BY-4.0 (components; see data/open/acyd_soy_brazil/README.md)"
ACYD_DATASET_ID = "acyd_soy_brazil_v1"
N_FEATURES = 37
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]
JOIN_KEYS = ("country", "admin_level_1", "admin_level_2", "year")
YIELD_FILE = "crop_soybean_yield.csv"
FEATURE_CHUNK_PATTERN = "features_chunk_{:03d}.csv"
FEATURE_CHUNK_COUNT = 100
SEASON_WEEKS = tuple(range(10, 41))

SOIL_COLUMNS = [
    "organic_carbon_0_5cm",
    "ph_h2o_0_5cm",
    "clay_0_5cm",
    "sand_0_5cm",
    "cec_0_5cm",
    "bulk_density_0_5cm",
]

WEATHER_PREFIXES = (
    "precipitation_week_",
    "t2m_min_week_",
    "t2m_max_week_",
    "solar_radiation_week_",
    "lai_high_week_",
    "ndvi_week_",
    "vapor_pressure_deficit_week_",
)


def acyd_hf_url(relative_path: str) -> str:
    """Return HuggingFace resolve URL for an ACYD repository path."""
    return f"{ACYD_HF_BASE}/{relative_path.lstrip('/')}"


def download_file(url: str, dest: Path) -> Path:
    """Download a remote file if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        urlretrieve(url, dest)  # noqa: S310 — pinned HuggingFace dataset URL
    return dest


def download_acyd_brazil_raw(
    raw_dir: Path,
    *,
    crop: str = "soybean",
    max_feature_chunks: int | None = None,
) -> dict[str, Path]:
    """Download ACYD Brazil raw CSVs (yield + feature chunks) into raw_dir."""
    if crop != "soybean":
        msg = f"only soybean is supported in v1, got {crop!r}"
        raise ValueError(msg)

    paths: dict[str, Path] = {}
    yield_rel = "brazil/final/crop/crop_soybean_yield.csv"
    yield_path = raw_dir / YIELD_FILE
    download_file(acyd_hf_url(yield_rel), yield_path)
    paths["yield"] = yield_path

    chunk_limit = FEATURE_CHUNK_COUNT if max_feature_chunks is None else min(
        max_feature_chunks,
        FEATURE_CHUNK_COUNT,
    )
    feature_dir = raw_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for idx in range(1, chunk_limit + 1):
        rel = f"brazil/final/features/{FEATURE_CHUNK_PATTERN.format(idx)}"
        dest = feature_dir / FEATURE_CHUNK_PATTERN.format(idx)
        download_file(acyd_hf_url(rel), dest)
        chunk_paths.append(dest)
    paths["feature_chunks"] = feature_dir
    paths["feature_chunk_files"] = chunk_paths  # type: ignore[assignment]
    return paths


def load_soybean_yield(yield_path: Path) -> pd.DataFrame:
    """Load and validate soybean yield table."""
    frame = pd.read_csv(yield_path)
    required = {"country", "admin_level_1", "admin_level_2", "year", "soybean_yield", "area_harvested"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"yield CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)
    frame = frame.dropna(subset=["soybean_yield", "area_harvested"])
    frame = frame[frame["area_harvested"] > 0]
    frame["year"] = frame["year"].astype(int)
    return frame.reset_index(drop=True)


def load_feature_chunks(chunk_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate ACYD feature chunk CSVs."""
    if not chunk_paths:
        msg = "no feature chunks provided"
        raise ValueError(msg)
    frames = [pd.read_csv(path) for path in sorted(chunk_paths)]
    combined = pd.concat(frames, ignore_index=True)
    combined["year"] = combined["year"].astype(int)
    return combined.drop_duplicates(subset=list(JOIN_KEYS), keep="last")


def _season_values(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    season_cols = [f"{prefix}{week}" for week in SEASON_WEEKS if f"{prefix}{week}" in frame.columns]
    if not season_cols:
        return np.full((len(frame), 1), np.nan, dtype=np.float64)
    return frame[season_cols].to_numpy(dtype=np.float64)


def _aggregate_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row mean, std, min, max ignoring NaNs."""
    n_rows = values.shape[0]
    mean = np.full(n_rows, np.nan, dtype=np.float64)
    std = np.full(n_rows, np.nan, dtype=np.float64)
    min_v = np.full(n_rows, np.nan, dtype=np.float64)
    max_v = np.full(n_rows, np.nan, dtype=np.float64)
    if values.size == 0 or values.shape[1] == 0:
        return mean, std, min_v, max_v

    valid = np.isfinite(values).any(axis=1)
    if not valid.any():
        return mean, std, min_v, max_v

    subset = values[valid]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean[valid] = np.nanmean(subset, axis=1)
        std[valid] = np.nanstd(subset, axis=1)
        min_v[valid] = np.nanmin(subset, axis=1)
        max_v[valid] = np.nanmax(subset, axis=1)
    return mean, std, min_v, max_v


def extract_acyd_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Build (n, 48) feature matrix from joined ACYD rows."""
    n = len(frame)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    col = 0

    out[:, col] = frame["latitude"].to_numpy(dtype=np.float32)
    col += 1
    out[:, col] = frame["longitude"].to_numpy(dtype=np.float32)
    col += 1
    out[:, col] = np.log1p(frame["area_harvested"].to_numpy(dtype=np.float64)).astype(np.float32)
    col += 1

    for soil_col in SOIL_COLUMNS:
        if soil_col not in frame.columns:
            msg = f"missing soil column: {soil_col}"
            raise ValueError(msg)
        out[:, col] = frame[soil_col].to_numpy(dtype=np.float32)
        col += 1

    for prefix in WEATHER_PREFIXES:
        values = _season_values(frame, prefix)
        mean, std, min_v, max_v = _aggregate_stats(values)
        out[:, col] = mean.astype(np.float32)
        col += 1
        out[:, col] = std.astype(np.float32)
        col += 1
        out[:, col] = min_v.astype(np.float32)
        col += 1
        out[:, col] = max_v.astype(np.float32)
        col += 1

    if col != N_FEATURES:
        msg = f"feature extractor produced {col} columns, expected {N_FEATURES}"
        raise ValueError(msg)
    return out


def build_binary_labels_below_state_median(yield_frame: pd.DataFrame) -> np.ndarray:
    """Label 1 when municipal yield is below state-year median."""
    grouped = yield_frame.groupby(["admin_level_1", "year"])["soybean_yield"]
    medians = grouped.transform("median")
    return (yield_frame["soybean_yield"] < medians).astype(np.int32).to_numpy()


def join_yield_features(yield_frame: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Inner join yield and climate/soil features."""
    merged = yield_frame.merge(feature_frame, on=list(JOIN_KEYS), how="inner", suffixes=("", "_feat"))
    if "latitude" not in merged.columns:
        msg = "joined frame missing latitude"
        raise ValueError(msg)
    return merged


def temporal_year_split(
    years: np.ndarray,
    *,
    train_max_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020, 2021),
    test_min_year: int = 2022,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for train / val / test by crop year."""
    train_mask = years <= train_max_year
    val_mask = np.isin(years, list(val_years))
    test_mask = years >= test_min_year
    return train_mask, val_mask, test_mask


def build_export_frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Build tabular_binary_v1 export frame."""
    frame = pd.DataFrame(features.astype(np.float32), columns=FEATURE_COLUMNS)
    frame[LABEL_COLUMN] = labels.astype(np.int32)
    return frame


def build_stats_payload(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    label_mode: str,
    train_max_year: int,
    val_years: tuple[int, ...],
    test_min_year: int,
) -> dict[str, Any]:
    """Build stats.json for ACYD temporal split."""

    def _split_stats(split_frame: pd.DataFrame) -> dict[str, Any]:
        pos = int((split_frame[LABEL_COLUMN] == 1).sum())
        neg = int((split_frame[LABEL_COLUMN] == 0).sum())
        total = len(split_frame)
        return {
            "n_rows": total,
            "class_counts": {"0": neg, "1": pos},
            "positive_rate": round(pos / total, 6) if total else 0.0,
        }

    return {
        "dataset_id": ACYD_DATASET_ID,
        "license": ACYD_LICENSE,
        "source_url": ACYD_SOURCE_URL,
        "n_features": N_FEATURES,
        "split_method": "temporal_crop_year",
        "label_mode": label_mode,
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
    }


def build_acyd_soy_processed(
    raw_dir: Path,
    out_dir: Path,
    *,
    label_mode: str = "below_state_median",
    train_max_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020, 2021),
    test_min_year: int = 2022,
    max_feature_chunks: int | None = None,
) -> dict[str, Path]:
    """Build parquet splits from downloaded ACYD raw files."""
    yield_path = raw_dir / YIELD_FILE
    feature_dir = raw_dir / "features"
    if not yield_path.is_file():
        msg = f"yield file missing: {yield_path}"
        raise FileNotFoundError(msg)

    chunk_paths = sorted(feature_dir.glob("features_chunk_*.csv"))
    if max_feature_chunks is not None:
        chunk_paths = chunk_paths[:max_feature_chunks]
    if not chunk_paths:
        msg = f"no feature chunks under {feature_dir}"
        raise FileNotFoundError(msg)

    yield_frame = load_soybean_yield(yield_path)
    feature_frame = load_feature_chunks(chunk_paths)
    merged = join_yield_features(yield_frame, feature_frame)

    if label_mode != "below_state_median":
        msg = f"unsupported label_mode: {label_mode}"
        raise ValueError(msg)

    labels = build_binary_labels_below_state_median(merged)
    features = extract_acyd_feature_matrix(merged)
    export = build_export_frame(features, labels)
    valid_mask = ~export.replace([np.inf, -np.inf], np.nan).isna().any(axis=1).to_numpy()
    export = export.loc[valid_mask].reset_index(drop=True)
    years = merged.loc[valid_mask, "year"].to_numpy(dtype=int)
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
        msg = (
            f"empty temporal split: train={len(train)} val={len(val)} test={len(test)} "
            f"(chunks={len(chunk_paths)})"
        )
        raise ValueError(msg)

    paths = write_parquet_splits(out_dir, train, val, test)
    stats = build_stats_payload(
        train,
        val,
        test,
        label_mode=label_mode,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
    )
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    paths["stats"] = stats_path
    return paths


def update_acyd_manifest_ready(
    manifest_path: Path,
    processed_dir: Path,
    *,
    dataset_id: str = ACYD_DATASET_ID,
) -> dict[str, Any]:
    """Mark ACYD dataset ready with checksums and row counts."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = next(d for d in manifest["datasets"] if d["id"] == dataset_id)
    files = dataset["files"]

    row_counts: dict[str, int] = {"total": 0}
    for split in ("train", "val", "test"):
        parquet_path = processed_dir / files[split]
        frame = pd.read_parquet(parquet_path)
        row_counts[split] = len(frame)
        row_counts["total"] += len(frame)

    checksums = {key: sha256_file(processed_dir / filename) for key, filename in files.items()}
    dataset["checksums"] = checksums
    dataset["row_counts"] = row_counts
    dataset["ready"] = True
    manifest["updated_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
