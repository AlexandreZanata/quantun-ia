#!/usr/bin/env python3
"""Run experiments with publication_large profile (n=1000, 10 seeds)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exp_id(run_py: Path) -> str:
    parts = run_py.parent.name.split("_")
    return f"{parts[0]}_{parts[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QML experiments with n=1000 profile")
    parser.add_argument(
        "--exp",
        nargs="*",
        metavar="EXP_ID",
        help="Experiment ids (e.g. exp_001). Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    runs = sorted((ROOT / "experiments").glob("exp_*/run.py"))
    if args.exp:
        allowed = set(args.exp)
        runs = [r for r in runs if _exp_id(r) in allowed]

    env = {**os.environ, "QML_PROFILE": "publication_large"}
    failures: list[str] = []

    for run_py in runs:
        exp_id = _exp_id(run_py)
        print(f"→ {exp_id} (profile=publication_large)")
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
