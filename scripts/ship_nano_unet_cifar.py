#!/usr/bin/env python3
"""CLI: ship NanoUNet CIFAR serve bundle (Phase K)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.application.image_nano_ship import ship_nano_unet_cifar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ci", "publication"), default="ci")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    ship_nano_unet_cifar(profile=args.profile, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
