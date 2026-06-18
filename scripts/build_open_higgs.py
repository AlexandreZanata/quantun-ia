#!/usr/bin/env python3
"""Build HIGGS open dataset — stratified 1.15M subsample to parquet splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.open_higgs import (
    HIGGS_URL,
    build_higgs_processed,
    download_higgs_raw,
    update_manifest_ready,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "open" / "higgs" / "raw" / "HIGGS.csv.gz"
DEFAULT_OUT = ROOT / "data" / "open" / "higgs" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HIGGS open dataset (Phase L1)")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="Raw HIGGS.csv.gz path")
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
        help="Do not update manifest.json (for dry runs)",
    )
    args = parser.parse_args()

    if args.skip_download and not args.raw.is_file():
        print(f"ERROR: raw file missing: {args.raw}")
        return 1

    if not args.skip_download:
        print(f"Downloading HIGGS from {HIGGS_URL} → {args.raw}")
        download_higgs_raw(args.raw)

    print(f"Processing HIGGS → {args.out}")
    paths = build_higgs_processed(args.raw, args.out)
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    if not args.skip_manifest:
        update_manifest_ready(args.manifest, args.out)
        print(f"Updated {args.manifest} (higgs_v1 ready=true)")

    print("HIGGS build complete.")
    print("Next: dvc add data/open/higgs/processed/v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
