#!/usr/bin/env python3
"""CLI — train, gate, publish, and export a shippable nanomodel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.application.nanomodel_registry import list_registry_keys
from src.application.nanomodel_ship import ShipNanomodelDTO, execute
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qml-ship",
        description="Train, gate, publish, and export a shippable nanomodel bundle",
    )
    parser.add_argument(
        "--model",
        required=True,
        dest="registry_key",
        help=f"Registry key (one of: {', '.join(list_registry_keys())})",
    )
    parser.add_argument("--profile", default=None, help="Override training profile")
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument("--retrain", action="store_true", help="Force retrain even if checkpoint exists")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; use existing checkpoint")
    parser.add_argument("--skip-gate", action="store_true", help="Skip real gate test (not for publication)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    outcome = execute(
        ShipNanomodelDTO(
            registry_key=args.registry_key,
            root=args.root,
            profile=args.profile,
            retrain=args.retrain,
            skip_train=args.skip_train,
            skip_gate=args.skip_gate,
        )
    )
    if isinstance(outcome, Fail):
        print(f"Error [{outcome.error.code}]: {outcome.error.message}", file=sys.stderr)
        return 1

    assert isinstance(outcome, Ok)
    result = outcome.value
    payload = {
        "registry_key": result.registry_key,
        "bundle_dir": result.bundle_dir,
        "serve_dir": result.serve_dir,
        "elapsed_s": result.elapsed_s,
        "stages": result.stages,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Shipped {result.registry_key}")
        print(f"Bundle: {result.bundle_dir}")
        print(f"Serve:  {result.serve_dir}")
        print(f"Elapsed: {result.elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
