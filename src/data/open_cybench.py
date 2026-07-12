"""CY-Bench maize slice (AgML sample US features) → open tabular binary dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.open_higgs import update_manifest_ready

CYBENCH_DATASET_ID = "cybench_maize_us_v1"
CYBENCH_LICENSE = "EUPL-1.2"
CYBENCH_SOURCE = "https://github.com/WUR-AI/sample_data (AgML CY-Bench sample; Zenodo DOI 10.5281/zenodo.11502142)"
CYBENCH_SAMPLE_REPO = "https://github.com/WUR-AI/sample_data.git"

# Official AgML sample train/test year cut (grain_maize_US_*.csv)
DEFAULT_TRAIN_MAX_YEAR = 2011
DEFAULT_VAL_YEARS = (2012, 2013, 2014, 2015)
DEFAULT_TEST_MIN_YEAR = 2016

META_COLS = ("adm_id", "year", "yield", "label")
# Yield lags / trend perfectly predict median-threshold labels — drop them.
LEAKY_FEATURE_PREFIXES = ("yield",)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync_cybench_sample_us(
    raw_dir: Path,
    *,
    sample_repo: Path | None = None,
) -> Path:
    """Copy AgML sample_data US maize feature CSVs into raw_dir/sample_us."""
    dest = raw_dir / "sample_us"
    dest.mkdir(parents=True, exist_ok=True)
    if sample_repo is None:
        # Prefer already-synced project raw files
        existing = list(dest.glob("grain_maize_US*.csv"))
        if len(existing) >= 2:
            return dest
        msg = (
            f"CY-Bench sample US CSVs missing under {dest}. "
            "Run: scripts/download_cybench_sample.py"
        )
        raise FileNotFoundError(msg)

    src = sample_repo / "features" / "maize" / "US"
    if not src.is_dir():
        msg = f"sample repo missing features/maize/US: {src}"
        raise FileNotFoundError(msg)
    for path in sorted(src.glob("grain_maize_US*.csv")):
        shutil.copy2(path, dest / path.name)
    readme = sample_repo / "README.md"
    if readme.is_file():
        shutil.copy2(readme, raw_dir / "SOURCE.md")
    return dest


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in META_COLS:
            continue
        if any(str(col).lower().startswith(prefix) for prefix in LEAKY_FEATURE_PREFIXES):
            continue
        cols.append(col)
    if not cols:
        msg = "no feature columns found in CY-Bench maize frame"
        raise ValueError(msg)
    return cols


def load_cybench_us_feature_table(sample_us_dir: Path) -> pd.DataFrame:
    """Load official train+test designed feature tables and concatenate."""
    train_path = sample_us_dir / "grain_maize_US_train.csv"
    test_path = sample_us_dir / "grain_maize_US_test.csv"
    if not train_path.is_file() or not test_path.is_file():
        msg = f"expected grain_maize_US_train/test.csv under {sample_us_dir}"
        raise FileNotFoundError(msg)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if list(train.columns) != list(test.columns):
        msg = "train/test CY-Bench feature schemas differ"
        raise ValueError(msg)
    frame = pd.concat([train, test], ignore_index=True)
    if frame["yield"].isna().any() or frame[list(_feature_columns(frame))].isna().any().any():
        msg = "CY-Bench sample contains NaN — refuse to build"
        raise ValueError(msg)
    return frame


def binarize_low_yield(
    frame: pd.DataFrame,
    *,
    train_max_year: int = DEFAULT_TRAIN_MAX_YEAR,
) -> tuple[pd.DataFrame, float]:
    """Label=1 when yield ≤ train-period median (low-yield risk)."""
    train_mask = frame["year"] <= int(train_max_year)
    if not train_mask.any():
        msg = f"no rows with year <= {train_max_year}"
        raise ValueError(msg)
    threshold = float(frame.loc[train_mask, "yield"].median())
    out = frame.copy()
    out["label"] = (out["yield"] <= threshold).astype(np.int32)
    return out, threshold


def split_cybench_years(
    frame: pd.DataFrame,
    *,
    train_max_year: int = DEFAULT_TRAIN_MAX_YEAR,
    val_years: tuple[int, ...] = DEFAULT_VAL_YEARS,
    test_min_year: int = DEFAULT_TEST_MIN_YEAR,
) -> dict[str, pd.DataFrame]:
    val_set = set(int(y) for y in val_years)
    splits = {
        "train": frame[frame["year"] <= int(train_max_year)].copy(),
        "val": frame[frame["year"].isin(val_set)].copy(),
        "test": frame[frame["year"] >= int(test_min_year)].copy(),
    }
    for name, part in splits.items():
        if part.empty:
            msg = f"CY-Bench split '{name}' is empty"
            raise ValueError(msg)
    return splits


def _to_feature_frame(part: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    data = {f"feature_{i}": part[col].to_numpy(dtype=np.float32) for i, col in enumerate(feature_cols)}
    data["label"] = part["label"].to_numpy(dtype=np.int32)
    return pd.DataFrame(data)


def build_cybench_maize_processed(
    sample_us_dir: Path,
    processed_dir: Path,
    *,
    train_max_year: int = DEFAULT_TRAIN_MAX_YEAR,
    val_years: tuple[int, ...] = DEFAULT_VAL_YEARS,
    test_min_year: int = DEFAULT_TEST_MIN_YEAR,
) -> dict[str, Any]:
    """Write train/val/test parquet + stats for cybench_maize_us_v1."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw = load_cybench_us_feature_table(sample_us_dir)
    feature_cols = _feature_columns(raw)
    labeled, threshold = binarize_low_yield(raw, train_max_year=train_max_year)
    splits = split_cybench_years(
        labeled,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
    )

    row_counts: dict[str, int] = {}
    split_stats: dict[str, Any] = {}
    for name, part in splits.items():
        frame = _to_feature_frame(part, feature_cols)
        out = processed_dir / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        n_rows = int(len(frame))
        n_pos = int(frame["label"].sum())
        row_counts[name] = n_rows
        split_stats[name] = {
            "n_rows": n_rows,
            "class_counts": {"0": n_rows - n_pos, "1": n_pos},
            "positive_rate": round(n_pos / n_rows, 6) if n_rows else 0.0,
        }

    stats = {
        "dataset_id": CYBENCH_DATASET_ID,
        "license": CYBENCH_LICENSE,
        "source": CYBENCH_SOURCE,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "low_yield_threshold": threshold,
        "split_method": "temporal_crop_year",
        "label_mode": "below_train_median",
        "temporal_split": {
            "train_max_year": int(train_max_year),
            "val_years": list(val_years),
            "test_min_year": int(test_min_year),
        },
        "train_max_year": int(train_max_year),
        "val_years": list(val_years),
        "test_min_year": int(test_min_year),
        "row_counts": {"total": int(sum(row_counts.values())), **row_counts},
        "splits": split_stats,
        "positive_rate": {name: split_stats[name]["positive_rate"] for name in ("train", "val", "test")},
    }
    stats_path = processed_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    stats["checksums"] = {
        name: _sha256_file(processed_dir / f"{name}.parquet")
        for name in ("train", "val", "test")
    }
    stats["checksums"]["stats"] = _sha256_file(stats_path)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def ensure_manifest_entry(manifest_path: Path) -> None:
    """Insert cybench_maize_us_v1 stub if missing (ready=false until build)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.setdefault("datasets", [])
    if any(d.get("id") == CYBENCH_DATASET_ID for d in datasets):
        return
    datasets.append(
        {
            "id": CYBENCH_DATASET_ID,
            "path": "cybench_maize/processed/v1",
            "description": (
                "CY-Bench maize US sample slice (AgML) — designed features, "
                "low-yield binary label, temporal splits"
            ),
            "license": CYBENCH_LICENSE,
            "source_url": CYBENCH_SOURCE,
            "ready": False,
            "row_counts": {"total": 0, "train": 0, "val": 0, "test": 0},
            "n_features": 0,
            "files": {
                "train": "train.parquet",
                "val": "val.parquet",
                "test": "test.parquet",
                "stats": "stats.json",
            },
            "checksums": {},
            "build_script": "scripts/build_open_cybench_maize.py",
        }
    )
    manifest["updated_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def update_cybench_manifest_ready(manifest_path: Path, processed_dir: Path) -> dict[str, Any]:
    ensure_manifest_entry(manifest_path)
    stats = json.loads((processed_dir / "stats.json").read_text(encoding="utf-8"))
    manifest = update_manifest_ready(
        manifest_path,
        processed_dir,
        dataset_id=CYBENCH_DATASET_ID,
    )
    dataset = next(d for d in manifest["datasets"] if d["id"] == CYBENCH_DATASET_ID)
    dataset["n_features"] = int(stats["n_features"])
    dataset["row_counts"] = stats["row_counts"]
    dataset["description"] = (
        "CY-Bench maize US sample slice (AgML) — designed features, "
        "low-yield binary label, temporal splits"
    )
    dataset["license"] = CYBENCH_LICENSE
    dataset["source_url"] = CYBENCH_SOURCE
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
