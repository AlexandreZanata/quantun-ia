"""Download and cache real datasets for Nano Parity Bench."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "data"


def _sklearn_ready(loader) -> str:
    loader()
    return "ready"


def _mnist_ready(cache_root: Path) -> str:
    root = cache_root / "raw" / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    datasets.MNIST(root=str(root), train=True, download=True)
    return "ready"


_DOWNLOADERS: dict[str, Any] = {
    "breast_cancer": lambda **_kw: _sklearn_ready(load_breast_cancer),
    "wine_binary": lambda **_kw: _sklearn_ready(load_wine),
    "iris_binary": lambda **_kw: _sklearn_ready(load_iris),
    "mnist_binary": lambda cache_root, **_kw: _mnist_ready(cache_root),
}


def ensure_datasets_available(
    dataset_names: list[str],
    *,
    cache_root: Path | None = None,
) -> dict[str, str]:
    """
    Ensure benchmark datasets are cached locally.

    - UCI tabular sets ship with scikit-learn (verified via load).
    - MNIST is downloaded via torchvision into data/raw/mnist.
    """
    root = cache_root or DEFAULT_CACHE_ROOT
    status: dict[str, str] = {}
    for name in dataset_names:
        downloader = _DOWNLOADERS.get(name)
        if downloader is None:
            status[name] = "unknown"
            continue
        status[name] = downloader(cache_root=root)
    return status
