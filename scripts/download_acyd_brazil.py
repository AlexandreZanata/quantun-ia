#!/usr/bin/env python3
"""Download ACYD Brazil raw CSVs from HuggingFace (soybean / maize)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_acyd import download_acyd_brazil_raw, normalize_crop

DEFAULT_OUT = {
    "soybean": ROOT / "data" / "open" / "acyd_soy_brazil" / "raw",
    "maize": ROOT / "data" / "open" / "acyd_maize_brazil" / "raw",
}
SOY_FEATURES = ROOT / "data" / "open" / "acyd_soy_brazil" / "raw" / "features"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ACYD Brazil raw data (HuggingFace)")
    parser.add_argument("--crop", default="soybean", choices=["soybean", "maize", "corn"])
    parser.add_argument("--out", type=Path, default=None, help="Raw output directory")
    parser.add_argument(
        "--max-feature-chunks",
        type=int,
        default=None,
        help="Limit feature chunks (default: all 100). Use 1-2 for smoke tests.",
    )
    parser.add_argument(
        "--reuse-features-from",
        type=Path,
        default=None,
        help="Reuse an existing features/ directory (default for maize: soy features if present)",
    )
    args = parser.parse_args()

    crop = normalize_crop(args.crop)
    out = args.out or DEFAULT_OUT[crop]
    reuse = args.reuse_features_from
    if reuse is None and crop == "maize" and SOY_FEATURES.is_dir():
        reuse = SOY_FEATURES

    print(f"Downloading ACYD Brazil ({crop}) → {out}")
    if args.max_feature_chunks:
        print(f"  feature chunks limited to {args.max_feature_chunks}")
    if reuse is not None:
        print(f"  reusing features from {reuse}")

    paths = download_acyd_brazil_raw(
        out,
        crop=crop,
        max_feature_chunks=args.max_feature_chunks,
        reuse_features_from=reuse,
    )
    n_chunks = len(paths.get("feature_chunk_files", []))
    print(f"  yield: {paths['yield']}")
    print(f"  feature chunks: {n_chunks} files under {paths['feature_chunks']}")
    print("Download complete.")
    if crop == "maize":
        print("Next: python scripts/build_open_acyd_maize.py")
    else:
        print("Next: python scripts/build_open_acyd_soy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
