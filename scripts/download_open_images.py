#!/usr/bin/env python3
"""Download open image packs for Cycle v3 nano I2I/T2I (Phase G).

P0 packs via torchvision (CIFAR-10, Fashion-MNIST, Oxford Flowers-102).
Prefers OSSCI / direct mirrors with wget -c when available (faster than default).
Writes under data/open/images/<pack>/raw/v1/ and a GENERATION sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ROOT = ROOT / "data" / "open" / "images"

PACKS = ("cifar10", "fashion_mnist", "flowers102")

# Prefer OSSCI / direct mirrors; fall back to origin.
CIFAR10_MIRRORS = (
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)
FLOWERS_BASE = "https://thor.robots.ox.ac.uk/flowers/102"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pack_dir(root: Path, name: str) -> Path:
    return root / name / "raw" / "v1"


def _download_url(url: str, dest: Path) -> None:
    """Download with wget -c when available, else urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        cmd = [
            "wget",
            "-c",
            "--progress=dot:giga",
            "-O",
            str(dest),
            url,
        ]
        subprocess.run(cmd, check=True)
        return
    print(f"  urllib ← {url}", flush=True)
    urlretrieve(url, dest)  # noqa: S310 — curated open-data URLs only


def download_cifar10(root: Path, *, force: bool) -> dict:
    from torchvision.datasets import CIFAR10

    from scripts.parallel_http_download import download as parallel_download

    dest = _pack_dir(root, "cifar10")
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "cifar10", "path": str(dest), "skipped": True}
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "cifar-10-python.tar.gz"
    extracted = dest / "cifar-10-batches-py"
    source = "local-cache"
    if force or not extracted.is_dir() or not archive.is_file() or archive.stat().st_size < 100_000_000:
        last_err: Exception | None = None
        for url in CIFAR10_MIRRORS:
            try:
                print(f"  parallel ← {url}", flush=True)
                parallel_download(url, archive, n_parts=8)
                source = url
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                print(f"  failed: {exc}", flush=True)
        else:
            msg = f"CIFAR-10 download failed: {last_err}"
            raise RuntimeError(msg)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=dest)
    checksum = _sha256_file(archive) if archive.is_file() else ""
    CIFAR10(root=str(dest), train=True, download=False)
    CIFAR10(root=str(dest), train=False, download=False)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "cifar10",
        "path": str(dest),
        "skipped": False,
        "n_train": 50000,
        "n_test": 10000,
        "source_url": source,
        "archive_sha256": checksum,
    }


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

    from scripts.parallel_http_download import download as parallel_download

    dest = _pack_dir(root, "flowers102")
    marker = dest / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "flowers102", "path": str(dest), "skipped": True}
    dest.mkdir(parents=True, exist_ok=True)
    flowers_dir = dest / "flowers-102"
    flowers_dir.mkdir(parents=True, exist_ok=True)
    for name in ("imagelabels.mat", "setid.mat"):
        target = flowers_dir / name
        if force or not target.is_file():
            _download_url(f"{FLOWERS_BASE}/{name}", target)
    archive = flowers_dir / "102flowers.tgz"
    jpg_dir = flowers_dir / "jpg"
    if force or not jpg_dir.is_dir() or not archive.is_file() or archive.stat().st_size < 100_000_000:
        print(f"  parallel ← {FLOWERS_BASE}/102flowers.tgz", flush=True)
        parallel_download(f"{FLOWERS_BASE}/102flowers.tgz", archive, n_parts=8)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=flowers_dir)
    for split in ("train", "val", "test"):
        Flowers102(root=str(dest), split=split, download=False)
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "flowers102",
        "path": str(dest),
        "skipped": False,
        "n_train": 1020,
        "n_val": 1020,
        "n_test": 6149,
        "source_url": FLOWERS_BASE,
        "archive_sha256": _sha256_file(archive),
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
