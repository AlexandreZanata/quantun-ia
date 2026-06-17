#!/usr/bin/env python3
"""Run experiments exp_011–exp_015 with a chosen profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_IDS = ["exp_011", "exp_012", "exp_013", "exp_014", "exp_015"]


def _exp_id(run_py: Path) -> str:
    parts = run_py.parent.name.split("_")
    return f"{parts[0]}_{parts[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exp_011–exp_015 batch")
    parser.add_argument(
        "--exp",
        nargs="*",
        default=DEFAULT_EXP_IDS,
        metavar="EXP_ID",
        help="Experiment ids (default: exp_011 … exp_015)",
    )
    parser.add_argument(
        "--profile",
        default="publication",
        help="Config profile (publication, ci, publication_large)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowed = set(args.exp)
    runs = sorted(
        r for r in (ROOT / "experiments").glob("exp_*/run.py") if _exp_id(r) in allowed
    )
    if not runs:
        print("No matching experiments found.", file=sys.stderr)
        return 1

    env = {**os.environ, "QML_PROFILE": args.profile, "MLFLOW_DISABLE": "1"}
    failures: list[str] = []

    for run_py in runs:
        exp_id = _exp_id(run_py)
        print(f"→ {exp_id} (profile={args.profile})")
        if args.dry_run:
            continue
        result = subprocess.run([sys.executable, str(run_py)], cwd=ROOT, env=env)
        if result.returncode != 0:
            failures.append(exp_id)
            print(f"  ✗ {exp_id} failed (exit {result.returncode})", file=sys.stderr)

    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
