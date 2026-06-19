#!/usr/bin/env python3
"""Export single consolidated model results file (all serve models + human demos)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.model_results_report import DEFAULT_REPORT_PATH, write_model_results_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export logs/model_results_summary.json")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    out = write_model_results_report(args.output, n_rows=args.rows, split=args.split)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
