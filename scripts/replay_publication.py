#!/usr/bin/env python3
"""One-command publication replay: large-profile runs plus export pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLICATION_STEPS: tuple[tuple[str, list[str]], ...] = (
    (
        "publication_large",
        [sys.executable, str(ROOT / "scripts" / "run_publication_large.py")],
    ),
    ("export_results", [sys.executable, str(ROOT / "scripts" / "export_results.py")]),
    ("figures", [sys.executable, str(ROOT / "scripts" / "generate_figures.py")]),
    (
        "latex_tables",
        [sys.executable, str(ROOT / "scripts" / "export_latex_tables.py")],
    ),
)

ARTIFACT_STEPS = PUBLICATION_STEPS[1:]


def replay_steps(*, artifacts_only: bool = False) -> tuple[tuple[str, list[str]], ...]:
    """Return ordered replay steps."""
    if artifacts_only:
        return ARTIFACT_STEPS
    return PUBLICATION_STEPS


def replay_publication(
    *,
    artifacts_only: bool = False,
    dry_run: bool = False,
    cwd: Path = ROOT,
) -> int:
    """Run publication replay pipeline. Returns process exit code (0 = success)."""
    env = {**os.environ, "MLFLOW_DISABLE": "1"}
    failures: list[str] = []

    for name, command in replay_steps(artifacts_only=artifacts_only):
        print(f"→ {name}")
        if dry_run:
            print(f"  {' '.join(command)}")
            continue
        result = subprocess.run(command, cwd=cwd, env=env)
        if result.returncode != 0:
            failures.append(name)
            print(f"  ✗ {name} failed (exit {result.returncode})", file=sys.stderr)

    if failures:
        print(f"Replay failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    mode = "artifacts-only" if artifacts_only else "full"
    print(f"Publication replay complete ({mode}).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay publication_large runs and regenerate export artifacts",
    )
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Skip experiment runs; regenerate CSV, figures, and LaTeX from logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()
    return replay_publication(artifacts_only=args.artifacts_only, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
