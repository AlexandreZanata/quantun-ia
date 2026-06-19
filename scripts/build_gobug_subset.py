#!/usr/bin/env python3
"""Build GoBug file-level defect open dataset from go-bug-collector."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.code_defects_gobug import (
    build_gobug_processed,
    download_gobug_raw,
    update_gobug_manifest_ready,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "open" / "code_defects_gobug" / "raw" / "combined"
DEFAULT_OUT = ROOT / "data" / "open" / "code_defects_gobug" / "processed" / "v1"
DEFAULT_MANIFEST = ROOT / "data" / "open" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GoBug file-level subset (exp_045)")
    parser.add_argument("--bug-csv", type=Path, default=DEFAULT_RAW / "file_bug_metrics.csv")
    parser.add_argument("--non-bug-csv", type=Path, default=DEFAULT_RAW / "file_non_bug_metrics.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()

    if args.skip_download and (not args.bug_csv.is_file() or not args.non_bug_csv.is_file()):
        print(f"ERROR: raw CSV missing under {DEFAULT_RAW}")
        return 1

    if not args.skip_download:
        print("Downloading GoBug combined CSVs…")
        download_gobug_raw(args.bug_csv, args.non_bug_csv)

    print(f"Processing GoBug → {args.out}")
    paths = build_gobug_processed(args.bug_csv, args.non_bug_csv, args.out)
    for name, path in paths.items():
        print(f"  wrote {name}: {path}")

    if not args.skip_manifest:
        update_gobug_manifest_ready(args.manifest, args.out)
        print(f"Updated {args.manifest} (code_defects_gobug_v1 ready=true)")

    print("GoBug build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
