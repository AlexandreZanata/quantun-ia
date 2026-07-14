"""Build processed split manifests for Cycle v3 open image packs (Phase G-T4).

Writes data/open/images/<pack>/processed/v1/stats.json with train/val/test
indices BEFORE any normalize/resize — loaders must honor these splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.open_images import PACK_LOADERS, is_pack_complete, pack_raw_dir

DEFAULT_ROOT = ROOT / "data" / "open" / "images"
SEED = 42


def _processed_dir(root: Path, pack: str) -> Path:
    return root / pack / "processed" / "v1"


def build_cifar10_splits(raw: Path, *, seed: int = SEED) -> dict:
    from torchvision.datasets import CIFAR10

    train_ds = CIFAR10(root=str(raw), train=True, download=False)
    test_ds = CIFAR10(root=str(raw), train=False, download=False)
    rng = np.random.default_rng(seed)
    n_train = len(train_ds)
    perm = rng.permutation(n_train)
    n_val = n_train // 10  # 10% holdout from train before any normalize
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    test_idx = list(range(len(test_ds)))
    return {
        "pack": "cifar10",
        "split_method": "random_train_holdout_before_normalize",
        "seed": seed,
        "n_features": None,
        "modality": "images",
        "spatial_shape": [32, 32, 3],
        "n_classes": 10,
        "splits": {
            "train": {"n": len(train_idx), "indices": train_idx},
            "val": {"n": len(val_idx), "indices": val_idx},
            "test": {"n": len(test_idx), "indices": test_idx},
        },
        "notes": "val carved from official train; official test untouched",
    }


def build_fashion_mnist_splits(raw: Path, *, seed: int = SEED) -> dict:
    from torchvision.datasets import FashionMNIST

    train_ds = FashionMNIST(root=str(raw), train=True, download=False)
    test_ds = FashionMNIST(root=str(raw), train=False, download=False)
    rng = np.random.default_rng(seed)
    n_train = len(train_ds)
    perm = rng.permutation(n_train)
    n_val = n_train // 10
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    test_idx = list(range(len(test_ds)))
    return {
        "pack": "fashion_mnist",
        "split_method": "random_train_holdout_before_normalize",
        "seed": seed,
        "modality": "images",
        "spatial_shape": [28, 28],
        "n_classes": 10,
        "splits": {
            "train": {"n": len(train_idx), "indices": train_idx},
            "val": {"n": len(val_idx), "indices": val_idx},
            "test": {"n": len(test_idx), "indices": test_idx},
        },
        "notes": "val carved from official train; official test untouched",
    }


def build_flowers102_splits(raw: Path, *, seed: int = SEED) -> dict:
    from torchvision.datasets import Flowers102

    # Official Oxford splits already defined — record sizes only (no reshuffle).
    train_ds = Flowers102(root=str(raw), split="train", download=False)
    val_ds = Flowers102(root=str(raw), split="val", download=False)
    test_ds = Flowers102(root=str(raw), split="test", download=False)
    return {
        "pack": "flowers102",
        "split_method": "official_oxford_setid",
        "seed": seed,
        "modality": "images",
        "spatial_shape": [64, 64, 3],
        "n_classes": 102,
        "splits": {
            "train": {"n": len(train_ds), "indices": list(range(len(train_ds)))},
            "val": {"n": len(val_ds), "indices": list(range(len(val_ds)))},
            "test": {"n": len(test_ds), "indices": list(range(len(test_ds)))},
        },
        "notes": "official train/val/test from setid.mat; resize deferred to loader",
    }


BUILDERS = {
    "cifar10": build_cifar10_splits,
    "fashion_mnist": build_fashion_mnist_splits,
    "flowers102": build_flowers102_splits,
}


def write_pack_stats(root: Path, pack: str, *, seed: int = SEED) -> Path:
    if not is_pack_complete(pack, root=root):
        msg = f"pack not complete: {pack}"
        raise FileNotFoundError(msg)
    raw = pack_raw_dir(pack, root=root)
    payload = BUILDERS[pack](raw, seed=seed)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_dir = _processed_dir(root, pack)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compact indices for large CIFAR/Fashion into npz; keep stats.json metadata slim.
    splits = payload["splits"]
    np.savez_compressed(
        out_dir / "split_indices.npz",
        train=np.asarray(splits["train"]["indices"], dtype=np.int64),
        val=np.asarray(splits["val"]["indices"], dtype=np.int64),
        test=np.asarray(splits["test"]["indices"], dtype=np.int64),
    )
    slim = dict(payload)
    slim["splits"] = {
        name: {"n": splits[name]["n"], "indices_file": "split_indices.npz"}
        for name in ("train", "val", "test")
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    return stats_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--packs", nargs="+", choices=PACK_LOADERS, default=list(PACK_LOADERS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    for pack in args.packs:
        print(f"Building splits for {pack} …", flush=True)
        path = write_pack_stats(args.root, pack, seed=args.seed)
        print(f"  → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
