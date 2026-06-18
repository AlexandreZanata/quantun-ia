#!/usr/bin/env python3
"""Export canonical reference datasets for Zenodo archival bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset_registry import get_dataset
from src.data.generators import load_synthetic_raw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "exports" / "reference_datasets"


def _write_csv_bundle(
    out_dir: Path,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_features = features.shape[1]
    columns = [f"feature_{i}" for i in range(n_features)]
    frame = pd.DataFrame(features, columns=columns)
    frame["label"] = labels.astype(int)
    csv_path = out_dir / f"{name}.csv"
    frame.to_csv(csv_path, index=False)
    meta_path = out_dir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return csv_path


def export_breast_cancer(out_dir: Path = DEFAULT_OUT) -> Path:
    """Export Wisconsin Breast Cancer (sklearn/UCI) features and binary labels."""
    features, labels, meta = get_dataset("breast_cancer", random_state=42)
    metadata = {
        "dataset": "breast_cancer",
        "source": "sklearn.datasets.load_breast_cancer",
        "license": "CC BY 4.0 (UCI ML Repository)",
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "label_meaning": "0=malignant, 1=benign",
        "preprocessing": "raw features — split and scale in experiment code only",
        "citation": "Wolberg, Street, Mangasarian — Wisconsin Breast Cancer Database",
        **meta,
    }
    return _write_csv_bundle(out_dir, "breast_cancer", features, labels, metadata)


def export_circles(out_dir: Path = DEFAULT_OUT, *, n_samples: int = 500, noise: float = 0.2) -> Path:
    """Export synthetic circles benchmark (publication default)."""
    features, labels, meta = load_synthetic_raw(
        n_samples=n_samples,
        dataset="circles",
        noise=noise,
        random_state=42,
    )
    metadata = {
        "dataset": "circles",
        "source": "sklearn.datasets.make_circles",
        "license": "N/A (synthetic)",
        "n_samples": n_samples,
        "noise": noise,
        "random_state": 42,
        "n_features": 2,
        "label_meaning": "binary class",
        "preprocessing": "raw 2D coordinates — scale after train/test split in experiments",
    }
    metadata.update(meta)
    metadata["source"] = "sklearn.datasets.make_circles"
    return _write_csv_bundle(out_dir, "circles", features, labels, metadata)


def export_all(out_dir: Path = DEFAULT_OUT) -> list[Path]:
    """Export all reference datasets for Zenodo bundles."""
    return [export_breast_cancer(out_dir), export_circles(out_dir)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reference datasets for Zenodo bundles")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: data/exports/reference_datasets)",
    )
    args = parser.parse_args()
    paths = export_all(args.out_dir)
    for path in paths:
        print(f"Wrote {path}")
    print(f"Exported {len(paths)} reference datasets to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
