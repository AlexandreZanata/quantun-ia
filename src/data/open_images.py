"""Load Cycle v3 open image packs downloaded under data/open/images/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
IMAGES_ROOT = ROOT_DEFAULT / "data" / "open" / "images"

PACK_LOADERS = ("cifar10", "fashion_mnist", "flowers102")
PACK_LOADERS_V4 = ("stl10", "tiny_imagenet")
ALL_IMAGE_PACKS = PACK_LOADERS + PACK_LOADERS_V4


def pack_raw_dir(pack: str, *, root: Path = IMAGES_ROOT) -> Path:
    return root / pack / "raw" / "v1"


def is_pack_complete(pack: str, *, root: Path = IMAGES_ROOT) -> bool:
    marker = pack_raw_dir(pack, root=root) / ".download_complete"
    return marker.is_file()


def _sample_indices(n_total: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    take = min(int(n_take), int(n_total))
    return rng.choice(n_total, take, replace=False)


def _load_tiny_imagenet_lists(raw: Path) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Return (train_paths, train_labels, val_paths, val_labels)."""
    root = raw / "tiny-imagenet-200"
    wnids = (root / "wnids.txt").read_text(encoding="utf-8").splitlines()
    class_to_idx = {w: i for i, w in enumerate(wnids)}
    train_paths: list[Path] = []
    train_labels: list[int] = []
    for wnid, idx in class_to_idx.items():
        img_dir = root / "train" / wnid / "images"
        for path in sorted(img_dir.glob("*.JPEG")):
            train_paths.append(path)
            train_labels.append(idx)
    val_paths: list[Path] = []
    val_labels: list[int] = []
    ann = root / "val" / "val_annotations.txt"
    for line in ann.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, wnid = parts[0], parts[1]
        val_paths.append(root / "val" / "images" / name)
        val_labels.append(class_to_idx[wnid])
    return train_paths, train_labels, val_paths, val_labels


def load_image_pack_arrays(
    pack: str,
    *,
    root: Path = IMAGES_ROOT,
    n_train: int = 8,
    n_test: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Load small numpy tensors from a completed torchvision / Tiny-IN pack (smoke / ci)."""
    if pack not in ALL_IMAGE_PACKS:
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

        train_idx = _sample_indices(len(train_ds), n_train, rng)
        test_idx = _sample_indices(len(test_ds), n_test, rng)
        x_train = np.stack([to_arr(train_ds[int(i)][0]) for i in train_idx])
        y_train = np.asarray([int(train_ds[int(i)][1]) for i in train_idx], dtype=np.int64)
        x_test = np.stack([to_arr(test_ds[int(i)][0]) for i in test_idx])
        y_test = np.asarray([int(test_ds[int(i)][1]) for i in test_idx], dtype=np.int64)
        n_train_available = len(train_ds)
        n_test_available = len(test_ds)

    elif pack == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        train_ds = FashionMNIST(root=str(dest), train=True, download=False)
        test_ds = FashionMNIST(root=str(dest), train=False, download=False)
        spatial_shape = (28, 28)

        def to_arr(img: Any) -> np.ndarray:
            return np.asarray(img, dtype=np.uint8)

        train_idx = _sample_indices(len(train_ds), n_train, rng)
        test_idx = _sample_indices(len(test_ds), n_test, rng)
        x_train = np.stack([to_arr(train_ds[int(i)][0]) for i in train_idx])
        y_train = np.asarray([int(train_ds[int(i)][1]) for i in train_idx], dtype=np.int64)
        x_test = np.stack([to_arr(test_ds[int(i)][0]) for i in test_idx])
        y_test = np.asarray([int(test_ds[int(i)][1]) for i in test_idx], dtype=np.int64)
        n_train_available = len(train_ds)
        n_test_available = len(test_ds)

    elif pack == "flowers102":
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
        n_train_available = len(train_ds)
        n_test_available = len(test_ds)

    elif pack == "stl10":
        from torchvision.datasets import STL10

        train_ds = STL10(root=str(dest), split="train", download=False)
        test_ds = STL10(root=str(dest), split="test", download=False)
        spatial_shape = (96, 96, 3)

        def to_arr(img: Any) -> np.ndarray:
            return np.asarray(img, dtype=np.uint8)

        train_idx = _sample_indices(len(train_ds), n_train, rng)
        test_idx = _sample_indices(len(test_ds), n_test, rng)
        x_train = np.stack([to_arr(train_ds[int(i)][0]) for i in train_idx])
        y_train = np.asarray([int(train_ds[int(i)][1]) for i in train_idx], dtype=np.int64)
        x_test = np.stack([to_arr(test_ds[int(i)][0]) for i in test_idx])
        y_test = np.asarray([int(test_ds[int(i)][1]) for i in test_idx], dtype=np.int64)
        n_train_available = len(train_ds)
        n_test_available = len(test_ds)

    else:  # tiny_imagenet
        from PIL import Image

        train_paths, train_labels, val_paths, val_labels = _load_tiny_imagenet_lists(dest)
        spatial_shape = (64, 64, 3)
        train_idx = _sample_indices(len(train_paths), n_train, rng)
        test_idx = _sample_indices(len(val_paths), n_test, rng)

        def load_resize(path: Path) -> np.ndarray:
            with Image.open(path) as img:
                return np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.uint8)

        x_train = np.stack([load_resize(train_paths[int(i)]) for i in train_idx])
        y_train = np.asarray([train_labels[int(i)] for i in train_idx], dtype=np.int64)
        x_test = np.stack([load_resize(val_paths[int(i)]) for i in test_idx])
        y_test = np.asarray([val_labels[int(i)] for i in test_idx], dtype=np.int64)
        n_train_available = len(train_paths)
        n_test_available = len(val_paths)

    return {
        "pack": pack,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "spatial_shape": spatial_shape,
        "n_train_available": n_train_available,
        "n_test_available": n_test_available,
    }


def summarize_packs(*, root: Path = IMAGES_ROOT, packs: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pack in packs or PACK_LOADERS:
        rows.append(
            {
                "pack": pack,
                "complete": is_pack_complete(pack, root=root),
                "raw_dir": str(pack_raw_dir(pack, root=root)),
            }
        )
    return rows


def pack_processed_dir(pack: str, *, root: Path = IMAGES_ROOT) -> Path:
    return root / pack / "processed" / "v1"


def load_split_indices(pack: str, *, root: Path = IMAGES_ROOT) -> dict[str, np.ndarray]:
    path = pack_processed_dir(pack, root=root) / "split_indices.npz"
    if not path.is_file():
        msg = f"missing split indices: {path}"
        raise FileNotFoundError(msg)
    data = np.load(path)
    return {k: data[k] for k in ("train", "val", "test")}


def _hwc_uint8_to_nchw_minus1_1(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32) / 255.0
    out = out.transpose(2, 0, 1)
    return out * 2.0 - 1.0


def load_cifar10_nchw(
    *,
    root: Path = IMAGES_ROOT,
    split: str = "train",
    n_take: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 images as float32 NCHW in [-1, 1] using Phase G split indices."""
    from torchvision.datasets import CIFAR10

    if split not in {"train", "val", "test"}:
        msg = f"invalid split: {split}"
        raise ValueError(msg)
    dest = pack_raw_dir("cifar10", root=root)
    if not is_pack_complete("cifar10", root=root):
        msg = "pack not downloaded: cifar10"
        raise FileNotFoundError(msg)

    indices = load_split_indices("cifar10", root=root)[split]
    # val carved from official train; test uses official test set
    train_flag = split != "test"
    ds = CIFAR10(root=str(dest), train=train_flag, download=False)

    rng = np.random.default_rng(seed)
    idx = np.asarray(indices, dtype=np.int64)
    if n_take is not None and n_take < len(idx):
        idx = rng.choice(idx, size=int(n_take), replace=False)

    images = []
    labels = []
    for i in idx:
        img, label = ds[int(i)]
        images.append(_hwc_uint8_to_nchw_minus1_1(np.asarray(img, dtype=np.uint8)))
        labels.append(int(label))
    x = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def load_pack_nchw(
    pack: str,
    *,
    root: Path = IMAGES_ROOT,
    split: str = "train",
    n_take: int | None = None,
    img_size: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Phase L/G pack as float32 NCHW in [-1, 1]; split indices before resize."""
    from PIL import Image
    from torchvision.datasets import STL10

    if pack not in ALL_IMAGE_PACKS:
        msg = f"unsupported pack: {pack}"
        raise ValueError(msg)
    if split not in {"train", "val", "test"}:
        msg = f"invalid split: {split}"
        raise ValueError(msg)
    if not is_pack_complete(pack, root=root):
        msg = f"pack not downloaded: {pack}"
        raise FileNotFoundError(msg)

    if pack == "cifar10":
        if img_size != 32:
            msg = "cifar10 NCHW loader only supports img_size=32"
            raise ValueError(msg)
        return load_cifar10_nchw(root=root, split=split, n_take=n_take, seed=seed)

    dest = pack_raw_dir(pack, root=root)
    indices = np.asarray(load_split_indices(pack, root=root)[split], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if n_take is not None and n_take < len(indices):
        indices = rng.choice(indices, size=int(n_take), replace=False)

    images: list[np.ndarray] = []
    labels: list[int] = []

    if pack == "stl10":
        # val/train carved from official train; test from official test
        stl_split = "test" if split == "test" else "train"
        ds = STL10(root=str(dest), split=stl_split, download=False)
        for i in indices:
            img, label = ds[int(i)]
            pil = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
            pil = pil.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            images.append(_hwc_uint8_to_nchw_minus1_1(np.asarray(pil, dtype=np.uint8)))
            labels.append(int(label))
    elif pack == "tiny_imagenet":
        train_paths, train_labels, val_paths, val_labels = _load_tiny_imagenet_lists(dest)
        if split == "test":
            paths, labs = val_paths, val_labels
        else:
            paths, labs = train_paths, train_labels
        for i in indices:
            with Image.open(paths[int(i)]) as img:
                pil = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                images.append(_hwc_uint8_to_nchw_minus1_1(np.asarray(pil, dtype=np.uint8)))
            labels.append(int(labs[int(i)]))
    else:
        msg = f"NCHW loader not implemented for pack={pack}"
        raise NotImplementedError(msg)

    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64)
