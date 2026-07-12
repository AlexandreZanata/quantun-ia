#!/usr/bin/env python3
"""Build cybench_maize_us_v1 processed parquet splits from AgML sample US features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_cybench import (
    CYBENCH_DATASET_ID,
    build_cybench_maize_processed,
    ensure_manifest_entry,
    update_cybench_manifest_ready,
)

DEFAULT_RAW = ROOT / "data" / "open" / "cybench_maize" / "raw" / "sample_us"
DEFAULT_PROCESSED = ROOT / "data" / "open" / "cybench_maize" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-us", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    ensure_manifest_entry(args.manifest)
    stats = build_cybench_maize_processed(args.sample_us, args.processed)
    update_cybench_manifest_ready(args.manifest, args.processed)
    print(f"Built {CYBENCH_DATASET_ID}")
    print(f"  features: {stats['n_features']}")
    print(f"  rows: {stats['row_counts']}")
    print(f"  low-yield threshold (train median): {stats['low_yield_threshold']:.4f}")
    print(f"  processed: {args.processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
