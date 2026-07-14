#!/usr/bin/env python3
"""Download Flickr8k captions + images for Cycle v3 G-T3 (T2I path).

Pokemon-blip is gated/DMCA — Flickr8k (jbrownlee mirror) is the P0 caption pack.
Writes under data/open/images/flickr8k/raw/v1/ and marks .download_complete.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.parallel_http_download import download as parallel_download

DEFAULT_ROOT = ROOT / "data" / "open" / "images" / "flickr8k" / "raw" / "v1"
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
IMG_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"


def download_flickr8k(raw: Path, *, force: bool = False) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".download_complete"
    if marker.is_file() and not force:
        return {"pack": "flickr8k", "path": str(raw), "skipped": True}

    text_zip = raw / "Flickr8k_text.zip"
    img_zip = raw / "Flickr8k_Dataset.zip"
    if force or not text_zip.is_file() or text_zip.stat().st_size < 1_000_000:
        print(f"  parallel ← {TEXT_URL}", flush=True)
        parallel_download(TEXT_URL, text_zip, n_parts=4)
    if force or not img_zip.is_file() or img_zip.stat().st_size < 100_000_000:
        print(f"  parallel ← {IMG_URL}", flush=True)
        parallel_download(IMG_URL, img_zip, n_parts=8)

    text_dir = raw / "Flickr8k_text"
    img_dir = raw / "Flicker8k_Dataset"
    if force or not text_dir.is_dir():
        with zipfile.ZipFile(text_zip) as zf:
            zf.extractall(raw)
    if force or not img_dir.is_dir() or len(list(img_dir.glob("*.jpg"))) < 1000:
        print("  extracting images (slow) …", flush=True)
        with zipfile.ZipFile(img_zip) as zf:
            zf.extractall(raw)

    n_jpg = len(list(img_dir.glob("*.jpg"))) if img_dir.is_dir() else 0
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "flickr8k",
        "path": str(raw),
        "n_images": n_jpg,
        "skipped": False,
        "source_text": TEXT_URL,
        "source_images": IMG_URL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    info = download_flickr8k(args.raw_dir, force=args.force)
    print(info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
