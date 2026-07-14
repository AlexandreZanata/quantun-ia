#!/usr/bin/env python3
"""Build COCO caption micro pairs.parquet + splits BEFORE resize (Phase L)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_higgs import sha256_file
from src.data.open_image_manifest import build_image_pack_manifest_entry, upsert_manifest_dataset

RAW_DEFAULT = ROOT / "data" / "open" / "images" / "coco_captions" / "raw" / "v1"
OUT_DEFAULT = ROOT / "data" / "open" / "images" / "coco_captions" / "processed" / "v1"
MANIFEST_PATH = ROOT / "data" / "open" / "manifest.json"
SEED = 42


def build_coco_caption_splits(
    raw: Path,
    out: Path,
    *,
    seed: int = SEED,
    register: bool = True,
) -> dict:
    marker = raw / ".download_complete"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    caps_raw = json.loads((raw / "captions_selected.json").read_text(encoding="utf-8"))
    caps = {str(k): str(v) for k, v in caps_raw.items()}
    ids_meta = json.loads((raw / "selected_image_ids.json").read_text(encoding="utf-8"))
    image_ids = [int(x) for x in ids_meta["image_ids"]]
    img_dir = raw / "train2017"

    rows = []
    for iid in image_ids:
        fname = f"{iid:012d}.jpg"
        path = img_dir / fname
        if not path.is_file():
            continue
        key = str(iid)
        if key not in caps:
            continue
        rows.append(
            {
                "image_id": key,
                "image_relpath": f"train2017/{fname}",
                "caption": caps[key],
            }
        )

    frame = pd.DataFrame(rows).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no COCO pairs found")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))
    n = len(frame)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    test_pos = set(perm[:n_test].tolist())
    val_pos = set(perm[n_test : n_test + n_val].tolist())
    splits = []
    for i in range(n):
        if i in test_pos:
            splits.append("test")
        elif i in val_pos:
            splits.append("val")
        else:
            splits.append("train")
    frame["split"] = splits

    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.parquet"
    frame.to_parquet(pairs_path, index=False)

    counts = {s: int((frame["split"] == s).sum()) for s in ("train", "val", "test")}
    np.savez_compressed(
        out / "split_indices.npz",
        train=np.where(frame["split"].to_numpy() == "train")[0].astype(np.int64),
        val=np.where(frame["split"].to_numpy() == "val")[0].astype(np.int64),
        test=np.where(frame["split"].to_numpy() == "test")[0].astype(np.int64),
    )
    stats = {
        "pack": "coco_captions",
        "split_method": "random_image_holdout_before_resize",
        "seed": seed,
        "modality": "image_text",
        "spatial_shape": [None, None, 3],
        "n_pairs": int(len(frame)),
        "splits": {k: {"n": v, "indices_file": "split_indices.npz"} for k, v in counts.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": "≤20k COCO 2017 train images; first caption per image; split before resize",
    }
    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    checksums = {
        "pairs": sha256_file(pairs_path),
        "stats": sha256_file(stats_path),
        "split_indices": sha256_file(out / "split_indices.npz"),
    }
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")

    if register:
        entry = build_image_pack_manifest_entry(
            dataset_id="coco_captions_micro_v1",
            description="COCO 2017 caption micro ≤20k images (Cycle v4 Phase L P0)",
            license_name="CC-BY-4.0",
            source_url="https://cocodataset.org/#download",
            spatial_shape=[None, None, 3],
            n_classes=None,
            modality="image_text",
            row_counts={
                "total": int(len(frame)),
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            },
            processed_rel="images/coco_captions/processed/v1",
            raw_rel="images/coco_captions/raw/v1",
            files={
                "pairs": "pairs.parquet",
                "stats": "stats.json",
                "split_indices": "split_indices.npz",
            },
            build_script="scripts/build_open_coco_caption_splits.py",
            root=ROOT,
        )
        upsert_manifest_dataset(MANIFEST_PATH, entry)
    return {"n_pairs": len(frame), "counts": counts, "out": str(out)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args(argv)
    info = build_coco_caption_splits(
        args.raw_dir, args.out_dir, seed=args.seed, register=not args.no_register
    )
    print(info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
