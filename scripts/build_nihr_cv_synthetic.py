#!/usr/bin/env python3
"""Build NIHR synthetic cardiovascular open dataset from Zenodo CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.nihr_cv_synthetic import (
    build_nihr_processed,
    download_nihr_raw,
    update_nihr_manifest_ready,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "open" / "nihr_cv_synthetic" / "raw" / "cvd_synthetic_dataset_v0.2.csv"
DEFAULT_OUT = ROOT / "data" / "open" / "nihr_cv_synthetic" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NIHR CV synthetic dataset (exp_044)")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Raw Zenodo CSV path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Processed output directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="manifest.json to update when build completes",
    )
    parser.add_argument("--skip-download", action="store_true", help="Require raw file to exist")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Do not update manifest.json",
    )
    args = parser.parse_args()

    if args.skip_download and not args.raw.is_file():
        print(f"ERROR: raw file missing: {args.raw}")
        return 1

    if not args.skip_download:
        print(f"Downloading NIHR CSV → {args.raw}")
        download_nihr_raw(args.raw)

    print(f"Processing NIHR CV → {args.out}")
    paths = build_nihr_processed(args.raw, args.out)
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    if not args.skip_manifest:
        update_nihr_manifest_ready(args.manifest, args.out)
        print(f"Updated {args.manifest} ({'nihr_cv_synthetic_v1'} ready=true)")

    print("NIHR CV build complete.")
    print("Next: dvc add data/open/nihr_cv_synthetic/processed/v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
