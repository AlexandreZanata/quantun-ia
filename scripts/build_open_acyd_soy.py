#!/usr/bin/env python3
"""Build ACYD Brazil soybean open dataset — temporal parquet splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_acyd import build_acyd_soy_processed, update_acyd_manifest_ready

DEFAULT_RAW = ROOT / "data" / "open" / "acyd_soy_brazil" / "raw"
DEFAULT_OUT = ROOT / "data" / "open" / "acyd_soy_brazil" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build acyd_soy_brazil_v1 (Phase L agro-climate)")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--label-mode",
        default="below_state_median",
        choices=["below_state_median"],
    )
    parser.add_argument("--train-max-year", type=int, default=2018)
    parser.add_argument("--test-min-year", type=int, default=2022)
    parser.add_argument(
        "--max-feature-chunks",
        type=int,
        default=None,
        help="Use subset of downloaded chunks (must match download step)",
    )
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    val_years = (2019, 2020, 2021)
    print(f"Building ACYD soybean → {args.out}")
    print(f"  temporal split: train ≤ {args.train_max_year}, val {val_years}, test ≥ {args.test_min_year}")

    paths = build_acyd_soy_processed(
        args.raw_dir,
        args.out,
        label_mode=args.label_mode,
        train_max_year=args.train_max_year,
        val_years=val_years,
        test_min_year=args.test_min_year,
        max_feature_chunks=args.max_feature_chunks,
    )
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    if not args.skip_manifest:
        update_acyd_manifest_ready(args.manifest, args.out)
        print(f"Updated {args.manifest} (acyd_soy_brazil_v1 ready=true)")

    print("ACYD build complete.")
    print("Next: dvc add data/open/acyd_soy_brazil/processed/v1")
    print("      make data-open-verify DATASET=acyd_soy_brazil_v1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
