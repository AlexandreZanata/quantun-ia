"""ACYD Brazil open dataset builder — crop yield + climate tabular features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.open_higgs import sha256_file, write_parquet_splits

ACYD_HF_REPO = "notadib/ACYD"
ACYD_HF_BASE = f"https://huggingface.co/datasets/{ACYD_HF_REPO}/resolve/main"
ACYD_SOURCE_URL = "https://huggingface.co/datasets/notadib/ACYD"
ACYD_LICENSE = "CC-BY-4.0 (components; see data/open/acyd_soy_brazil/README.md)"
ACYD_DATASET_ID = "acyd_soy_brazil_v1"
ACYD_MAIZE_DATASET_ID = "acyd_maize_brazil_v1"
N_FEATURES = 37
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]
JOIN_KEYS = ("country", "admin_level_1", "admin_level_2", "year")
YIELD_FILE = "crop_soybean_yield.csv"
FEATURE_CHUNK_PATTERN = "features_chunk_{:03d}.csv"
FEATURE_CHUNK_COUNT = 100
SEASON_WEEKS = tuple(range(10, 41))

# maize aliases to ACYD's corn yield file
CROP_SPECS: dict[str, dict[str, str]] = {
    "soybean": {
        "yield_file": "crop_soybean_yield.csv",
        "yield_column": "soybean_yield",
        "hf_yield_rel": "brazil/final/crop/crop_soybean_yield.csv",
        "dataset_id": ACYD_DATASET_ID,
    },
    "maize": {
        "yield_file": "crop_corn_yield.csv",
        "yield_column": "corn_yield",
        "hf_yield_rel": "brazil/final/crop/crop_corn_yield.csv",
        "dataset_id": ACYD_MAIZE_DATASET_ID,
    },
}
CROP_ALIASES = {"corn": "maize", "soy": "soybean"}


def normalize_crop(crop: str) -> str:
    """Map crop aliases to canonical crop keys."""
    key = CROP_ALIASES.get(crop, crop)
    if key not in CROP_SPECS:
        msg = f"unsupported crop {crop!r}; expected one of {sorted(CROP_SPECS)}"
        raise ValueError(msg)
    return key


def crop_spec(crop: str) -> dict[str, str]:
    """Return ACYD crop specification for a canonical or aliased crop name."""
    return CROP_SPECS[normalize_crop(crop)]


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


def _link_or_copy_features(source_dir: Path, dest_dir: Path) -> None:
    """Reuse climate feature chunks via symlink when possible."""
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists() or dest_dir.is_symlink():
        return
    try:
        dest_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
    except OSError:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_dir.glob("features_chunk_*.csv")):
            target = dest_dir / path.name
            if not target.exists():
                target.write_bytes(path.read_bytes())


def download_acyd_brazil_raw(
    raw_dir: Path,
    *,
    crop: str = "soybean",
    max_feature_chunks: int | None = None,
    reuse_features_from: Path | None = None,
) -> dict[str, Path]:
    """Download ACYD Brazil raw CSVs (yield + feature chunks) into raw_dir."""
    spec = crop_spec(crop)
    paths: dict[str, Path] = {}
    yield_path = raw_dir / spec["yield_file"]
    download_file(acyd_hf_url(spec["hf_yield_rel"]), yield_path)
    paths["yield"] = yield_path

    feature_dir = raw_dir / "features"
    if reuse_features_from is not None:
        source = Path(reuse_features_from)
        if not source.is_dir():
            msg = f"reuse_features_from is not a directory: {source}"
            raise FileNotFoundError(msg)
        _link_or_copy_features(source, feature_dir)
        chunk_paths = sorted(feature_dir.glob("features_chunk_*.csv"))
        if max_feature_chunks is not None:
            chunk_paths = chunk_paths[:max_feature_chunks]
    else:
        chunk_limit = FEATURE_CHUNK_COUNT if max_feature_chunks is None else min(
            max_feature_chunks,
            FEATURE_CHUNK_COUNT,
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        for idx in range(1, chunk_limit + 1):
            rel = f"brazil/final/features/{FEATURE_CHUNK_PATTERN.format(idx)}"
            dest = feature_dir / FEATURE_CHUNK_PATTERN.format(idx)
            download_file(acyd_hf_url(rel), dest)
            chunk_paths.append(dest)

    paths["feature_chunks"] = feature_dir
    paths["feature_chunk_files"] = chunk_paths  # type: ignore[assignment]
    return paths


def load_crop_yield(yield_path: Path, *, crop: str = "soybean") -> pd.DataFrame:
    """Load and validate an ACYD crop yield table."""
    spec = crop_spec(crop)
    yield_column = spec["yield_column"]
    frame = pd.read_csv(yield_path)
    required = {
        "country",
        "admin_level_1",
        "admin_level_2",
        "year",
        yield_column,
        "area_harvested",
    }
    missing = required - set(frame.columns)
    if missing:
        msg = f"yield CSV missing columns: {sorted(missing)}"
        raise ValueError(msg)
    frame = frame.dropna(subset=[yield_column, "area_harvested"])
    frame = frame[frame["area_harvested"] > 0]
    frame["year"] = frame["year"].astype(int)
    return frame.reset_index(drop=True)


def load_soybean_yield(yield_path: Path) -> pd.DataFrame:
    """Load and validate soybean yield table."""
    return load_crop_yield(yield_path, crop="soybean")


def load_maize_yield(yield_path: Path) -> pd.DataFrame:
    """Load and validate maize (corn) yield table."""
    return load_crop_yield(yield_path, crop="maize")


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


def _resolve_yield_column(yield_frame: pd.DataFrame, yield_column: str | None) -> str:
    if yield_column is not None:
        if yield_column not in yield_frame.columns:
            msg = f"yield column missing: {yield_column}"
            raise ValueError(msg)
        return yield_column
    for candidate in ("soybean_yield", "corn_yield"):
        if candidate in yield_frame.columns:
            return candidate
    msg = "yield frame has no recognized yield column (soybean_yield / corn_yield)"
    raise ValueError(msg)


def build_binary_labels_below_state_median(
    yield_frame: pd.DataFrame,
    *,
    yield_column: str | None = None,
) -> np.ndarray:
    """Label 1 when municipal yield is below state-year median."""
    column = _resolve_yield_column(yield_frame, yield_column)
    grouped = yield_frame.groupby(["admin_level_1", "year"])[column]
    medians = grouped.transform("median")
    return (yield_frame[column] < medians).astype(np.int32).to_numpy()


def count_season_weeks_above(
    frame: pd.DataFrame,
    prefix: str,
    threshold: float,
    *,
    weeks: tuple[int, ...] = SEASON_WEEKS,
) -> np.ndarray:
    """Count in-season weeks where ``prefix{week}`` exceeds ``threshold``."""
    cols = [f"{prefix}{week}" for week in weeks if f"{prefix}{week}" in frame.columns]
    if not cols:
        return np.zeros(len(frame), dtype=np.int32)
    values = frame[cols].to_numpy(dtype=np.float64)
    return np.sum(values > threshold, axis=1).astype(np.int32)


def count_season_weeks_below(
    frame: pd.DataFrame,
    prefix: str,
    threshold: float,
    *,
    weeks: tuple[int, ...] = SEASON_WEEKS,
) -> np.ndarray:
    """Count in-season weeks where ``prefix{week}`` is below ``threshold``."""
    cols = [f"{prefix}{week}" for week in weeks if f"{prefix}{week}" in frame.columns]
    if not cols:
        return np.zeros(len(frame), dtype=np.int32)
    values = frame[cols].to_numpy(dtype=np.float64)
    return np.sum(values < threshold, axis=1).astype(np.int32)


def seasonal_total_precipitation(
    frame: pd.DataFrame,
    *,
    weeks: tuple[int, ...] = SEASON_WEEKS,
) -> np.ndarray:
    """Sum in-season weekly precipitation (mm) per row."""
    cols = [f"precipitation_week_{week}" for week in weeks if f"precipitation_week_{week}" in frame.columns]
    if not cols:
        return np.full(len(frame), np.nan, dtype=np.float64)
    values = frame[cols].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.nansum(values, axis=1)


def build_compound_stress_labels(
    merged: pd.DataFrame,
    *,
    heat_threshold_k: float = 308.15,
    min_hot_weeks: int = 3,
    drought_precip_z: float = -1.0,
    drought_precip_train_mask: np.ndarray | None = None,
    drought_weekly_mm: float = 5.0,
    min_dry_weeks: int = 4,
) -> np.ndarray:
    """
    Label 1 when yield is below state-year median AND (drought OR heat stress).

    Drought uses seasonal precipitation z-score (SPEI proxy) when a train mask is
    supplied; otherwise falls back to counting dry weeks below ``drought_weekly_mm``.
    Heat uses weekly ``t2m_max`` above ``heat_threshold_k`` (35 °C default).
    """
    low_yield = build_binary_labels_below_state_median(merged).astype(bool)
    hot_weeks = count_season_weeks_above(merged, "t2m_max_week_", heat_threshold_k)
    heat_stress = hot_weeks >= min_hot_weeks

    season_precip = seasonal_total_precipitation(merged)
    if drought_precip_train_mask is not None and np.any(drought_precip_train_mask):
        train_totals = season_precip[drought_precip_train_mask]
        train_totals = train_totals[np.isfinite(train_totals)]
        if train_totals.size >= 2:
            mean = float(np.mean(train_totals))
            std = float(np.std(train_totals))
            if std > 0:
                z_scores = (season_precip - mean) / std
                drought_stress = z_scores <= drought_precip_z
            else:
                drought_stress = count_season_weeks_below(
                    merged, "precipitation_week_", drought_weekly_mm
                ) >= min_dry_weeks
        else:
            drought_stress = count_season_weeks_below(
                merged, "precipitation_week_", drought_weekly_mm
            ) >= min_dry_weeks
    else:
        drought_stress = count_season_weeks_below(
            merged, "precipitation_week_", drought_weekly_mm
        ) >= min_dry_weeks

    climate_stress = drought_stress | heat_stress
    return (low_yield & climate_stress).astype(np.int32)


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
    dataset_id: str = ACYD_DATASET_ID,
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
        "dataset_id": dataset_id,
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


def build_acyd_crop_processed(
    raw_dir: Path,
    out_dir: Path,
    *,
    crop: str = "soybean",
    label_mode: str = "below_state_median",
    train_max_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020, 2021),
    test_min_year: int = 2022,
    max_feature_chunks: int | None = None,
) -> dict[str, Path]:
    """Build parquet splits from downloaded ACYD raw files for a crop."""
    spec = crop_spec(crop)
    yield_path = raw_dir / spec["yield_file"]
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

    yield_frame = load_crop_yield(yield_path, crop=crop)
    feature_frame = load_feature_chunks(chunk_paths)
    merged = join_yield_features(yield_frame, feature_frame)

    if label_mode != "below_state_median":
        msg = f"unsupported label_mode: {label_mode}"
        raise ValueError(msg)

    labels = build_binary_labels_below_state_median(
        merged, yield_column=spec["yield_column"]
    )
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
        dataset_id=spec["dataset_id"],
    )
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    paths["stats"] = stats_path
    return paths


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
    """Build parquet splits from downloaded ACYD soybean raw files."""
    return build_acyd_crop_processed(
        raw_dir,
        out_dir,
        crop="soybean",
        label_mode=label_mode,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
        max_feature_chunks=max_feature_chunks,
    )


def build_acyd_maize_processed(
    raw_dir: Path,
    out_dir: Path,
    *,
    label_mode: str = "below_state_median",
    train_max_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020, 2021),
    test_min_year: int = 2022,
    max_feature_chunks: int | None = None,
) -> dict[str, Path]:
    """Build parquet splits from downloaded ACYD maize (corn) raw files."""
    return build_acyd_crop_processed(
        raw_dir,
        out_dir,
        crop="maize",
        label_mode=label_mode,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
        max_feature_chunks=max_feature_chunks,
    )


def load_acyd_compound_stress_splits(
    root: Path,
    *,
    n_train_rows: int | None = None,
    n_val_rows: int | None = None,
    random_state: int = 42,
    train_max_year: int = 2018,
    val_years: tuple[int, ...] = (2019, 2020, 2021),
    test_min_year: int = 2022,
    heat_threshold_k: float = 308.15,
    min_hot_weeks: int = 3,
    drought_precip_z: float = -1.0,
    drought_weekly_mm: float = 5.0,
    min_dry_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Load ACYD features with compound-stress labels and train-only scaling."""
    raw_dir = root / "data" / "open" / "acyd_soy_brazil" / "raw"
    yield_path = raw_dir / YIELD_FILE
    feature_dir = raw_dir / "features"
    if not yield_path.is_file():
        msg = f"yield file missing: {yield_path}"
        raise FileNotFoundError(msg)
    chunk_paths = sorted(feature_dir.glob("features_chunk_*.csv"))
    if not chunk_paths:
        msg = f"no feature chunks under {feature_dir}"
        raise FileNotFoundError(msg)

    yield_frame = load_soybean_yield(yield_path)
    feature_frame = load_feature_chunks(chunk_paths)
    merged = join_yield_features(yield_frame, feature_frame)
    features = extract_acyd_feature_matrix(merged)
    years = merged["year"].to_numpy(dtype=int)
    train_mask, val_mask, test_mask = temporal_year_split(
        years,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
    )
    labels = build_compound_stress_labels(
        merged,
        heat_threshold_k=heat_threshold_k,
        min_hot_weeks=min_hot_weeks,
        drought_precip_z=drought_precip_z,
        drought_precip_train_mask=train_mask,
        drought_weekly_mm=drought_weekly_mm,
        min_dry_weeks=min_dry_weeks,
    )

    valid = ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    features = features[valid]
    labels = labels[valid]
    years = years[valid]
    train_mask = train_mask[valid]
    val_mask = val_mask[valid]
    test_mask = test_mask[valid]

    x_train, y_train = features[train_mask], labels[train_mask].astype(np.float32)
    x_val, y_val = features[val_mask], labels[val_mask].astype(np.float32)
    x_test, y_test = features[test_mask], labels[test_mask].astype(np.float32)

    if n_train_rows is not None and n_train_rows < len(y_train):
        idx = np.arange(len(y_train))
        selected, _ = train_test_split(
            idx,
            train_size=n_train_rows,
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
            random_state=random_state,
        )
        x_train, y_train = x_train[selected], y_train[selected]

    if n_val_rows is not None and n_val_rows < len(y_val):
        idx = np.arange(len(y_val))
        selected, _ = train_test_split(
            idx,
            train_size=n_val_rows,
            stratify=y_val if len(np.unique(y_val)) > 1 else None,
            random_state=random_state,
        )
        x_val, y_val = x_val[selected], y_val[selected]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return x_train, y_train, x_val, y_val, x_test, y_test, scaler


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
