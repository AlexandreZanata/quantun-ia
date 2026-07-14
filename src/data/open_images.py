"""Load Cycle v3 open image packs downloaded under data/open/images/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = ROOT_DEFAULT / "data" / "open" / "images"

PACK_LOADERS = ("cifar10", "fashion_mnist", "flowers102")


def pack_raw_dir(pack: str, *, root: Path = IMAGES_ROOT) -> Path:
    return root / pack / "raw" / "v1"


def is_pack_complete(pack: str, *, root: Path = IMAGES_ROOT) -> bool:
    marker = pack_raw_dir(pack, root=root) / ".download_complete"
    return marker.is_file()


def _sample_indices(n_total: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    take = min(int(n_take), int(n_total))
    return rng.choice(n_total, take, replace=False)


def load_image_pack_arrays(
    pack: str,
    *,
    root: Path = IMAGES_ROOT,
    n_train: int = 8,
    n_test: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Load small numpy tensors from a completed torchvision pack (smoke / ci)."""
    if pack not in PACK_LOADERS:
        msg = f"unsupported pack: {pack}"
        raise ValueError(msg)
    dest = pack_raw_dir(pack, root=root)
    if not is_pack_complete(pack, root=root):
        msg = f"pack not downloaded: {pack} (missing {dest / '.download_complete'})"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)

    if pack == "cifar10":
        from torchvision.datasets import CIFAR10

        train_ds = CIFAR10(root=str(dest), train=True, download=False)
        test_ds = CIFAR10(root=str(dest), train=False, download=False)
        spatial_shape: tuple[int, ...] = (32, 32, 3)

        def to_arr(img: Any) -> np.ndarray:
            return np.asarray(img, dtype=np.uint8)

    elif pack == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        train_ds = FashionMNIST(root=str(dest), train=True, download=False)
        test_ds = FashionMNIST(root=str(dest), train=False, download=False)
        spatial_shape = (28, 28)

        def to_arr(img: Any) -> np.ndarray:
            return np.asarray(img, dtype=np.uint8)

    else:
        from torchvision.datasets import Flowers102

        train_ds = Flowers102(root=str(dest), split="train", download=False)
        test_ds = Flowers102(root=str(dest), split="test", download=False)
        spatial_shape = (64, 64, 3)

        def to_arr(img: Any) -> np.ndarray:
            return np.asarray(img.resize((64, 64)), dtype=np.uint8)

    train_idx = _sample_indices(len(train_ds), n_train, rng)
    test_idx = _sample_indices(len(test_ds), n_test, rng)
    x_train = np.stack([to_arr(train_ds[int(i)][0]) for i in train_idx])
    y_train = np.asarray([int(train_ds[int(i)][1]) for i in train_idx], dtype=np.int64)
    x_test = np.stack([to_arr(test_ds[int(i)][0]) for i in test_idx])
    y_test = np.asarray([int(test_ds[int(i)][1]) for i in test_idx], dtype=np.int64)

    return {
        "pack": pack,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "spatial_shape": spatial_shape,
        "n_train_available": len(train_ds),
        "n_test_available": len(test_ds),
    }


def summarize_packs(*, root: Path = IMAGES_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pack in PACK_LOADERS:
        rows.append(
            {
                "pack": pack,
                "complete": is_pack_complete(pack, root=root),
                "raw_dir": str(pack_raw_dir(pack, root=root)),
            }
        )
    return rows
