#!/usr/bin/env python3
"""Download AgML CY-Bench sample_data (maize US designed features)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_cybench import CYBENCH_SAMPLE_REPO, sync_cybench_sample_us

DEFAULT_RAW = ROOT / "data" / "open" / "cybench_maize" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--sample-repo",
        type=Path,
        default=None,
        help="Existing checkout of WUR-AI/sample_data (otherwise shallow clone to temp)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if sample_us CSVs already exist",
    )
    args = parser.parse_args()

    dest_us = args.raw_dir / "sample_us"
    if (
        not args.force
        and (dest_us / "grain_maize_US_train.csv").is_file()
        and (dest_us / "grain_maize_US_test.csv").is_file()
    ):
        print(f"CY-Bench sample US already present → {dest_us}")
        return 0

    if args.sample_repo is not None:
        dest = sync_cybench_sample_us(args.raw_dir, sample_repo=args.sample_repo)
        print(f"Synced CY-Bench sample US → {dest}")
        return 0

    with tempfile.TemporaryDirectory(prefix="cybench_sample_") as tmp:
        tmp_path = Path(tmp) / "sample_data"
        print(f"Cloning {CYBENCH_SAMPLE_REPO} (depth=1) …")
        subprocess.run(  # noqa: S603
            ["git", "clone", "--depth", "1", CYBENCH_SAMPLE_REPO, str(tmp_path)],
            check=True,
        )
        dest = sync_cybench_sample_us(args.raw_dir, sample_repo=tmp_path)
        print(f"Synced CY-Bench sample US → {dest}")
        src_readme = tmp_path / "README.md"
        if src_readme.is_file():
            shutil.copy2(src_readme, args.raw_dir / "SOURCE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
