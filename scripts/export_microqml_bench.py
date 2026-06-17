#!/usr/bin/env python3
"""Export MicroQML Bench v1 JSON bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark.microqml_bench import build_bench_export, write_bench_export

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MicroQML Bench v1 bundle")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "microqml_bench" / "v1.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    export = build_bench_export()
    path = write_bench_export(export, args.output)
    print(f"MicroQML Bench v1 exported → {path} ({len(export['leaderboard'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
