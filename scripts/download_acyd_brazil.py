#!/usr/bin/env python3
"""Download ACYD Brazil raw CSVs from HuggingFace (soybean v1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_acyd import download_acyd_brazil_raw

DEFAULT_OUT = ROOT / "data" / "open" / "acyd_soy_brazil" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ACYD Brazil raw data (HuggingFace)")
    parser.add_argument("--crop", default="soybean", choices=["soybean"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Raw output directory")
    parser.add_argument(
        "--max-feature-chunks",
        type=int,
        default=None,
        help="Limit feature chunks (default: all 100). Use 1-2 for smoke tests.",
    )
    args = parser.parse_args()

    print(f"Downloading ACYD Brazil ({args.crop}) → {args.out}")
    if args.max_feature_chunks:
        print(f"  feature chunks limited to {args.max_feature_chunks}")

    paths = download_acyd_brazil_raw(
        args.out,
        crop=args.crop,
        max_feature_chunks=args.max_feature_chunks,
    )
    n_chunks = len(paths.get("feature_chunk_files", []))
    print(f"  yield: {paths['yield']}")
    print(f"  feature chunks: {n_chunks} files under {paths['feature_chunks']}")
    print("Download complete.")
    print("Next: python scripts/build_open_acyd_soy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
