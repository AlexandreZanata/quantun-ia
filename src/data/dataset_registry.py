"""Unified dataset registry for synthetic and real-world loaders."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.data import real_datasets
from src.data.generators import load_synthetic_raw
from src.data.splits import split_train_test

REAL_LOADERS = {
    "breast_cancer": real_datasets.load_breast_cancer_raw,
    "wine_binary": real_datasets.load_wine_binary_raw,
    "iris_binary": real_datasets.load_iris_binary_raw,
}

SYNTHETIC_DATASETS = {"moons", "circles", "classification"}


def get_dataset(name: str, random_state: int = 42, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load raw (unscaled) features and binary labels.

    For synthetic 2D sets, delegates to ``load_synthetic_raw``.
    For real tabular sets, returns sklearn/torchvision features without scaling.
    """
    if name in SYNTHETIC_DATASETS:
        return load_synthetic_raw(
            dataset=name,
            n_samples=kwargs.get("n_samples", 200),
            noise=kwargs.get("noise", 0.1),
            random_state=random_state,
        )
    if name == "mnist_binary":
        return real_datasets.load_mnist_binary_raw(
            n_samples=kwargs.get("n_samples", 500),
            class_a=kwargs.get("class_a", 0),
            class_b=kwargs.get("class_b", 1),
            random_state=random_state,
        )
    if name == "sequential_binary":
        return real_datasets.make_sequential_binary(
            n_samples=kwargs.get("n_samples", 200),
            seq_len=kwargs.get("seq_len", 8),
            input_dim=kwargs.get("input_dim", 2),
            noise=kwargs.get("noise", 0.1),
            random_state=random_state,
        )
    if name == "sequential_phase":
        return real_datasets.make_sequential_phase(
            n_samples=kwargs.get("n_samples", 200),
            seq_len=kwargs.get("seq_len", 12),
            input_dim=kwargs.get("input_dim", 4),
            noise=kwargs.get("noise", 0.15),
            random_state=random_state,
        )
    if name not in REAL_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    return REAL_LOADERS[name]()


def prepare_dataset(
    name: str,
    *,
    random_state: int = 42,
    test_size: float = 0.3,
    scale: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Split and optionally scale (train-fit only) for tabular or 2D synthetic data."""
    if name == "mnist_binary":
        return real_datasets.prepare_mnist_pca_splits(
            n_samples=kwargs.get("n_samples", 500),
            n_components=kwargs.get("n_components", 8),
            test_size=test_size,
            random_state=random_state,
            class_a=kwargs.get("class_a", 0),
            class_b=kwargs.get("class_b", 1),
        )

    if name == "sequential_binary":
        return real_datasets.prepare_sequential_splits(
            n_samples=kwargs.get("n_samples", 200),
            seq_len=kwargs.get("seq_len", 8),
            input_dim=kwargs.get("input_dim", 2),
            noise=kwargs.get("noise", 0.1),
            test_size=test_size,
            random_state=random_state,
        )

    if name == "sequential_phase":
        X, y, meta = real_datasets.make_sequential_phase(
            n_samples=kwargs.get("n_samples", 200),
            seq_len=kwargs.get("seq_len", 12),
            input_dim=kwargs.get("input_dim", 4),
            noise=kwargs.get("noise", 0.15),
            random_state=random_state,
        )
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=test_size, random_state=random_state
        )
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
        X_train = ((X_train - mean) / std).astype(np.float32)
        X_test = ((X_test - mean) / std).astype(np.float32)
        meta.update({"seq_len": kwargs.get("seq_len", 12), "input_dim": kwargs.get("input_dim", 4)})
        return X_train, X_test, y_train, y_test, meta

    X, y, meta = get_dataset(name, random_state=random_state, **kwargs)
    X_train, X_test, y_train, y_test, split_meta = real_datasets.prepare_tabular_splits(
        X, y, test_size=test_size, random_state=random_state, scale=scale
    )
    meta.update(split_meta)
    return X_train, X_test, y_train, y_test, meta
