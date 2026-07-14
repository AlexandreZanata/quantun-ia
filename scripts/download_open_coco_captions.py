#!/usr/bin/env python3
"""Download COCO 2017 caption micro pack (≤20k images) for Cycle v4 Phase L.

Downloads captions_train2017 annotations, samples ≤max_images unique image IDs,
then fetches individual train2017 JPEGs (avoids the full 18 GB zip).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.parallel_http_download import download as parallel_download

DEFAULT_RAW = ROOT / "data" / "open" / "images" / "coco_captions" / "raw" / "v1"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_TMPL = "http://images.cocodataset.org/train2017/{stem:012d}.jpg"


def _download_one(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return
    urlretrieve(url, dest)  # noqa: S310 — curated COCO URLs


def download_coco_captions_micro(
    raw: Path,
    *,
    max_images: int = 20_000,
    seed: int = 42,
    workers: int = 16,
    force: bool = False,
) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / ".download_complete"
    img_dir = raw / "train2017"
    ann_json = raw / "annotations" / "captions_train2017.json"
    if marker.is_file() and ann_json.is_file() and img_dir.is_dir() and not force:
        n = len(list(img_dir.glob("*.jpg")))
        return {"pack": "coco_captions", "path": str(raw), "skipped": True, "n_images": n}

    ann_zip = raw / "annotations_trainval2017.zip"
    if force or not ann_json.is_file():
        print(f"  parallel ← {ANN_URL}", flush=True)
        parallel_download(ANN_URL, ann_zip, n_parts=4)
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extractall(raw)

    payload = json.loads(ann_json.read_text(encoding="utf-8"))
    id_to_file = {int(img["id"]): str(img["file_name"]) for img in payload["images"]}
    captions_by_id: dict[int, str] = {}
    for row in payload["annotations"]:
        iid = int(row["image_id"])
        if iid not in captions_by_id:
            captions_by_id[iid] = str(row["caption"]).strip()

    rng = np.random.default_rng(seed)
    eligible = sorted(set(id_to_file) & set(captions_by_id))
    take = min(int(max_images), len(eligible))
    chosen = sorted(rng.choice(eligible, size=take, replace=False).tolist())

    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw / "selected_image_ids.json"
    meta_path.write_text(json.dumps({"image_ids": chosen, "seed": seed}, indent=2) + "\n", encoding="utf-8")

    # First caption map for selected ids
    cap_map = {iid: captions_by_id[iid] for iid in chosen}
    (raw / "captions_selected.json").write_text(json.dumps(cap_map, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for iid in chosen:
        fname = id_to_file[iid]
        dest = img_dir / fname
        url = IMG_TMPL.format(stem=iid)
        jobs.append((url, dest))

    print(f"  fetching {len(jobs)} COCO train2017 JPEGs (workers={workers}) …", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_download_one, url, dest): (url, dest) for url, dest in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(jobs)}", flush=True)
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print(f"  download fail: {exc}", flush=True)

    n_jpg = len(list(img_dir.glob("*.jpg")))
    if n_jpg < take * 0.9:
        msg = f"COCO micro download incomplete: got {n_jpg}/{take}"
        raise RuntimeError(msg)

    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return {
        "pack": "coco_captions",
        "path": str(raw),
        "skipped": False,
        "n_images": n_jpg,
        "max_images": take,
        "failed": failed,
        "source_annotations": ANN_URL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--max-images", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    info = download_coco_captions_micro(
        args.raw_dir,
        max_images=args.max_images,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
    )
    print(info, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
