"""Load Flickr8k caption pairs for Cycle v3 T2I (G-T3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
PACK_ROOT = ROOT_DEFAULT / "data" / "open" / "images" / "flickr8k"
RAW_DIR = PACK_ROOT / "raw" / "v1"
PROCESSED_DIR = PACK_ROOT / "processed" / "v1"


def is_flickr8k_ready(*, root: Path = PACK_ROOT) -> bool:
    raw = root / "raw" / "v1"
    processed = root / "processed" / "v1"
    return (raw / ".download_complete").is_file() and (processed / "pairs.parquet").is_file()


def load_caption_frame(*, processed: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = processed / "pairs.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_flickr8k_batch(
    split: str = "train",
    *,
    n_take: int = 32,
    img_size: int = 32,
    seed: int = 42,
    root: Path = PACK_ROOT,
    null_captions: bool = False,
) -> dict[str, Any]:
    """Load NCHW float images in [-1, 1] + captions. Split assigned before resize."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    processed = root / "processed" / "v1"
    raw = root / "raw" / "v1"
    frame = load_caption_frame(processed=processed)
    subset = frame[frame["split"] == split].reset_index(drop=True)
    if subset.empty:
        raise RuntimeError(f"no rows for split={split}")
    rng = np.random.default_rng(seed)
    take = min(int(n_take), len(subset))
    idx = rng.choice(len(subset), take, replace=False)
    rows = subset.iloc[idx]

    images: list[np.ndarray] = []
    captions: list[str] = []
    for _, row in rows.iterrows():
        path = raw / str(row["image_relpath"])
        img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # HWC [-1,1]
        images.append(np.transpose(arr, (2, 0, 1)))
        captions.append("" if null_captions else str(row["caption"]))
    x = np.stack(images, axis=0)
    return {
        "pack": "flickr8k",
        "split": split,
        "images": x,
        "captions": captions,
        "n_available": int(len(subset)),
        "img_size": img_size,
    }
