#!/usr/bin/env python3
"""Push DVC-tracked publication artifacts to configured remote."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.dvc_remote_setup import dvc_module_available, resolve_dvc_cmd, setup_local_remote

ROOT = Path(__file__).resolve().parents[1]


def push_artifacts(*, replay: bool = False, root: Path = ROOT) -> None:
    if replay:
        subprocess.run(
            [sys.executable, "scripts/replay_publication.py", "--artifacts-only"],
            cwd=root,
            check=True,
        )
    dvc_cmd = resolve_dvc_cmd()
    subprocess.run([*dvc_cmd, "push"], cwd=root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Push DVC artifacts to configured remote")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Regenerate exports/figures/tables before push",
    )
    parser.add_argument(
        "--setup-remote",
        action="store_true",
        help="Ensure local filesystem remote exists before push",
    )
    args = parser.parse_args()

    if not dvc_module_available():
        print(
            "ERROR: DVC not installed. Run: make dvc-setup",
            file=sys.stderr,
        )
        return 1

    try:
        if args.setup_remote:
            setup_local_remote(root=ROOT)
        push_artifacts(replay=args.replay, root=ROOT)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed (exit {exc.returncode})", file=sys.stderr)
        return exc.returncode or 1

    print("DVC push complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
