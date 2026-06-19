#!/usr/bin/env python3
"""CLI — install a shippable nanomodel bundle into artifacts/ for inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.application.model_export import install_bundle_to_artifacts
from src.application.nanomodel_registry import get_nanomodel_spec, list_registry_keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qml-download",
        description="Install a shippable nanomodel bundle from dist/serve_models/",
    )
    parser.add_argument(
        "--model",
        required=True,
        dest="registry_key",
        help=f"Registry key (one of: {', '.join(list_registry_keys())})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing dist/serve_models/",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args(argv)

    try:
        spec = get_nanomodel_spec(args.registry_key)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    bundle_dir = args.root / spec.bundle_dir
    if not (bundle_dir / "best.pt").is_file():
        print(
            f"Bundle not found: {bundle_dir}\n"
            f"Run: qml-ship --model {args.registry_key} --skip-train --skip-gate",
            file=sys.stderr,
        )
        return 1

    target = install_bundle_to_artifacts(spec, bundle_dir)
    payload = {"registry_key": spec.registry_key, "installed_to": str(target)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Installed {spec.registry_key} → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
