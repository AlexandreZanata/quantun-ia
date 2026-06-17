#!/usr/bin/env python3
"""CLI entry point for running quantun-ia experiments."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qml-run",
        description="Run a quantun-ia experiment script with optional profile override",
    )
    parser.add_argument(
        "run_script",
        type=Path,
        help="Path to experiment run.py (e.g. experiments/exp_001_quantum_vs_classical/run.py)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Config profile override (ci, publication, publication_large)",
    )
    parsed_argv = [str(a) for a in argv] if argv is not None else None
    args = parser.parse_args(parsed_argv)

    script = args.run_script.resolve()
    if not script.exists():
        print(f"Error: script not found: {script}", file=sys.stderr)
        return 1

    if args.profile:
        os.environ["QML_PROFILE"] = args.profile

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.chdir(ROOT)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
