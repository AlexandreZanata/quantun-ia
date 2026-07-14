#!/usr/bin/env python3
"""Export Cycle v3 paper tables (Phase K / K-T4)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.cycle_v3_tables import export_cycle_v3_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/tables"),
        help="Output directory for .tex tables",
    )
    args = parser.parse_args(argv)
    written = export_cycle_v3_tables(out_dir=args.out_dir)
    for name, path in written.items():
        print(f"wrote {path} ({name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
