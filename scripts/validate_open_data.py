#!/usr/bin/env python3
"""Validate Phase L open data manifest, checksums, schema, and DVC pointer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.open_manifest import validate_open_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate open data manifest (Phase L2 gate)")
    parser.add_argument(
        "--dataset",
        default="higgs_v1",
        help="Dataset id in manifest.json (default: higgs_v1)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: project root)",
    )
    args = parser.parse_args()

    ok, issues = validate_open_data(args.root, dataset_id=args.dataset)
    for item in issues:
        print(f"ERROR: {item}", file=sys.stderr)

    if ok:
        print(f"Open data validation passed ({args.dataset}).")
        return 0

    print("Open data validation failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
