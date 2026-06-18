#!/usr/bin/env python3
"""Validate Phase L open data manifest, checksums, schema, and DVC pointer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.open_manifest import validate_all_ready_open_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate open data manifest (Phase L2 gate)")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset id in manifest.json (default: all ready datasets)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: project root)",
    )
    args = parser.parse_args()

    if args.dataset:
        from src.data.open_manifest import validate_open_data

        ok, issues = validate_open_data(args.root, dataset_id=args.dataset)
        label = args.dataset
    else:
        ok, issues = validate_all_ready_open_data(args.root)
        label = "all ready datasets"

    for item in issues:
        print(f"ERROR: {item}", file=sys.stderr)

    if ok:
        print(f"Open data validation passed ({label}).")
        return 0

    print("Open data validation failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
