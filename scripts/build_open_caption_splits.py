#!/usr/bin/env python3
"""Build Flickr8k caption pairs parquet + splits BEFORE any resize (G-T3).

Uses official Flickr_8k.{train,dev,test}Images.txt lists when present.
Writes data/open/images/flickr8k/processed/v1/{pairs.parquet,stats.json,split_indices.npz}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DEFAULT = ROOT / "data" / "open" / "images" / "flickr8k" / "raw" / "v1"
OUT_DEFAULT = ROOT / "data" / "open" / "images" / "flickr8k" / "processed" / "v1"
SEED = 42


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_split_ids(text_dir: Path) -> dict[str, set[str]]:
    mapping = {
        "train": "Flickr_8k.trainImages.txt",
        "val": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }
    out: dict[str, set[str]] = {}
    for split, name in mapping.items():
        path = text_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        }
        out[split] = ids
    return out


def _parse_token_captions(token_path: Path) -> dict[str, list[str]]:
    """Parse Flickr8k.token.txt → image_id -> list of captions (strip #n)."""
    caps: dict[str, list[str]] = {}
    for line in token_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        key, caption = line.split("\t", 1)
        image_id = key.split("#", 1)[0].strip()
        caption = caption.strip()
        if not image_id or not caption:
            continue
        caps.setdefault(image_id, []).append(caption)
    return caps


def build_flickr8k_pairs(
    raw: Path,
    out: Path,
    *,
    max_captions_per_image: int = 1,
    seed: int = SEED,
) -> dict:
    marker = raw / ".download_complete"
    if not marker.is_file():
        raise FileNotFoundError(f"missing download marker: {marker} (run download_open_captions.py)")

    text_dir = raw / "Flickr8k_text"
    img_dir = raw / "Flicker8k_Dataset"
    token_path = text_dir / "Flickr8k.token.txt"
    if not token_path.is_file():
        # some extracts nest one level
        nested = list(raw.glob("**/Flickr8k.token.txt"))
        if not nested:
            raise FileNotFoundError(token_path)
        token_path = nested[0]
        text_dir = token_path.parent

    caps = _parse_token_captions(token_path)
    splits = _load_split_ids(text_dir)
    id_to_split: dict[str, str] = {}
    for split, ids in splits.items():
        for image_id in ids:
            id_to_split[image_id] = split

    rows: list[dict] = []
    missing = 0
    for image_id, captions in sorted(caps.items()):
        split = id_to_split.get(image_id)
        if split is None:
            continue
        img_path = img_dir / image_id
        if not img_path.is_file():
            missing += 1
            continue
        for caption in captions[:max_captions_per_image]:
            rows.append(
                {
                    "image_id": image_id,
                    "image_relpath": f"Flicker8k_Dataset/{image_id}",
                    "caption": caption,
                    "split": split,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no caption pairs built — check extract paths")

    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.parquet"
    frame.to_parquet(pairs_path, index=False)

    # Stable integer indices per split (row index into pairs.parquet)
    split_indices: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        idx = np.asarray(frame.index[frame["split"] == split].tolist(), dtype=np.int64)
        split_indices[split] = idx
    np.savez_compressed(out / "split_indices.npz", **split_indices)

    stats = {
        "pack": "flickr8k",
        "dataset_id": "flickr8k_captions_v1",
        "modality": "image_text",
        "split_method": "official_flickr8k_lists_before_resize",
        "seed": seed,
        "max_captions_per_image": max_captions_per_image,
        "spatial_shape": [None, None, 3],
        "n_images_missing": missing,
        "row_counts": {
            "total": int(len(frame)),
            "train": int((frame["split"] == "train").sum()),
            "val": int((frame["split"] == "val").sum()),
            "test": int((frame["split"] == "test").sum()),
        },
        "n_unique_images": int(frame["image_id"].nunique()),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Splits assigned from official Flickr_8k.*Images.txt before any resize/normalize",
    }
    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    checksums = {
        "pairs": _sha256_file(pairs_path),
        "stats": _sha256_file(stats_path),
        "split_indices": _sha256_file(out / "split_indices.npz"),
    }
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    return {**stats, "checksums": checksums, "out": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-captions-per-image", type=int, default=1)
    args = parser.parse_args()
    info = build_flickr8k_pairs(
        args.raw_dir,
        args.out_dir,
        max_captions_per_image=args.max_captions_per_image,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "out"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
