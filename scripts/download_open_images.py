#!/usr/bin/env python3
"""Download open image packs for Cycle v3 nano I2I/T2I (Phase G).

P0 packs via torchvision (CIFAR-10, Fashion-MNIST, Oxford Flowers-102).
Writes under data/open/images/<pack>/raw/v1/ and a GENERATION sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data" / "open" / "images"

PACKS = ("cifar10", "fashion_mnist", "flowers102")


def _sha256_file(path: Path) -> str:  # noqa: ARG001 — reserved for future checksum manifests
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack_dir(root: Path, name: str) -> Path:
    return root / name / "raw" / "v1"


def download_cifar10(root: Path, *, force: bool) -> dict:
    from torchvision.datasets import CIFAR10

    dest = _pack_dir(root, "cifar10")
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "cifar10", "path": str(dest), "skipped": True}
    dest.mkdir(parents=True, exist_ok=True)
    CIFAR10(root=str(dest), train=True, download=True)
    CIFAR10(root=str(dest), train=False, download=True)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {"pack": "cifar10", "path": str(dest), "skipped": False, "n_train": 50000, "n_test": 10000}


def download_fashion_mnist(root: Path, *, force: bool) -> dict:
    from torchvision.datasets import FashionMNIST

    dest = _pack_dir(root, "fashion_mnist")
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "fashion_mnist", "path": str(dest), "skipped": True}
    dest.mkdir(parents=True, exist_ok=True)
    FashionMNIST(root=str(dest), train=True, download=True)
    FashionMNIST(root=str(dest), train=False, download=True)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "fashion_mnist",
        "path": str(dest),
        "skipped": False,
        "n_train": 60000,
        "n_test": 10000,
    }


def download_flowers102(root: Path, *, force: bool) -> dict:
    from torchvision.datasets import Flowers102

    dest = _pack_dir(root, "flowers102")
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "flowers102", "path": str(dest), "skipped": True}
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        Flowers102(root=str(dest), split=split, download=True)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "flowers102",
        "path": str(dest),
        "skipped": False,
        "n_train": 1020,
        "n_val": 1020,
        "n_test": 6149,
    }


DOWNLOADERS = {
    "cifar10": download_cifar10,
    "fashion_mnist": download_fashion_mnist,
    "flowers102": download_flowers102,
}


def write_generation_md(root: Path, results: list[dict]) -> Path:
    path = root / "GENERATION.md"
    lines = [
        "# Open image packs — generation log",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "**Script:** `scripts/download_open_images.py`",
        "",
        "## License / source matrix (P0)",
        "",
        "| Pack | Source | License notes |",
        "|------|--------|---------------|",
        "| cifar10 | Toronto CIFAR / torchvision | Research use; cite Krizhevsky 2009 |",
        "| fashion_mnist | Zalando Research / torchvision | MIT |",
        "| flowers102 | Oxford VGG / torchvision | Research use; cite Nilsback & Zisserman 2008 |",
        "",
        "## Downloads",
        "",
    ]
    for row in results:
        lines.append(f"- `{row['pack']}` → `{row['path']}` (skipped={row.get('skipped')})")
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Raw blobs under `*/raw/v1/` — gitignored / DVC later",
            "- Train/val/test **split before** normalize for experiment `run.py`",
            "- Caption packs (Flickr8k, pokemon-blip) are Phase G-T3 — separate script",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_stats(root: Path, results: list[dict]) -> Path:
    path = root / "download_stats.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "packs": results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Destination root (default: data/open/images)",
    )
    parser.add_argument(
        "--packs",
        nargs="+",
        choices=PACKS,
        default=list(PACKS),
        help="Which P0 packs to download",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if complete")
    args = parser.parse_args(argv)

    args.root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for name in args.packs:
        print(f"Downloading {name} …", flush=True)
        results.append(DOWNLOADERS[name](args.root, force=args.force))
        print(f"  → {results[-1]}", flush=True)

    gen = write_generation_md(args.root, results)
    stats = write_stats(args.root, results)
    print(f"Wrote {gen}")
    print(f"Wrote {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
