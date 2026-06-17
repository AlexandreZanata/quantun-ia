"""Real-world dataset loaders (raw features, no scaling)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from torchvision import datasets

from src.data.scaling import pca_train_test, scale_train_test
from src.data.splits import split_train_test


def load_breast_cancer_raw() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    return X, y, {"name": "breast_cancer", "n_features": X.shape[1], "source": "sklearn"}


def load_wine_binary_raw() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_wine()
    mask = data.target < 2
    X = data.data[mask].astype(np.float32)
    y = data.target[mask].astype(np.float32)
    y = (y > 0).astype(np.float32)
    return X, y, {"name": "wine_binary", "n_features": X.shape[1], "source": "sklearn"}


def load_iris_binary_raw() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = load_iris()
    mask = data.target != 1
    X = data.data[mask].astype(np.float32)
    y = data.target[mask].astype(np.float32)
    y = (y > 0).astype(np.float32)
    return X, y, {"name": "iris_binary", "n_features": X.shape[1], "source": "sklearn"}


def load_mnist_binary_raw(
    n_samples: int = 500,
    class_a: int = 0,
    class_b: int = 1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = datasets.MNIST(root="data/raw/mnist", train=True, download=True)
    images, labels = dataset.data.numpy(), dataset.targets.numpy()
    mask = (labels == class_a) | (labels == class_b)
    X_all = images[mask].reshape(-1, 784).astype(np.float32) / 255.0
    y_all = (labels[mask] == class_b).astype(np.float32)

    rng = np.random.default_rng(random_state)
    if n_samples < len(X_all):
        idx = rng.choice(len(X_all), size=n_samples, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]

    return X_all, y_all, {
        "name": "mnist_binary",
        "n_features": X_all.shape[1],
        "class_a": class_a,
        "class_b": class_b,
        "source": "torchvision",
    }


def make_sequential_binary(
    n_samples: int = 200,
    seq_len: int = 8,
    input_dim: int = 2,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Synthetic sequence classification (GRU/Transformer-friendly)."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, (n_samples, seq_len, input_dim)).astype(np.float32)
    if noise > 0:
        X = X + rng.normal(0, noise, X.shape).astype(np.float32)
    signal = X[:, :, 0].sum(axis=1) + 0.5 * X[:, :, 1].sum(axis=1)
    y = (signal > 0).astype(np.float32)
    return X, y, {
        "name": "sequential_binary",
        "seq_len": seq_len,
        "input_dim": input_dim,
        "source": "synthetic",
    }


def make_sequential_phase(
    n_samples: int = 200,
    seq_len: int = 12,
    input_dim: int = 4,
    noise: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Phase-sensitive sequence task — label depends on temporal ordering.

    PCA on flattened windows loses phase progression; transformers / fusion should help.
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, seq_len, dtype=np.float32)
    X = np.zeros((n_samples, seq_len, input_dim), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        freq = rng.uniform(0.8, 1.6)
        phase0 = rng.uniform(0, 2 * np.pi)
        for d in range(input_dim):
            offset = d * np.pi / max(input_dim, 1)
            X[i, :, d] = np.sin(freq * t + phase0 + offset).astype(np.float32)
        if noise > 0:
            X[i] += rng.normal(0, noise, (seq_len, input_dim)).astype(np.float32)
        first_half = X[i, : seq_len // 2, :].sum()
        second_half = X[i, seq_len // 2 :, :].sum()
        y[i] = float(second_half > first_half)

    return X, y, {
        "name": "sequential_phase",
        "seq_len": seq_len,
        "input_dim": input_dim,
        "source": "synthetic_phase",
    }


def prepare_mnist_pca_splits(
    n_samples: int,
    n_components: int,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    class_a: int = 0,
    class_b: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X, y, meta = load_mnist_binary_raw(
        n_samples=n_samples,
        class_a=class_a,
        class_b=class_b,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test, pca = pca_train_test(X_train, X_test, n_components, random_state=random_state)
    meta.update({"pca_components": n_components, "pca": pca})
    return X_train, X_test, y_train, y_test, meta


def prepare_tabular_splits(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.3,
    random_state: int = 42,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    meta: dict[str, Any] = {"scaled": scale}
    if scale:
        X_train, X_test, scaler = scale_train_test(X_train, X_test)
        meta["scaler"] = scaler
    return X_train, X_test, y_train, y_test, meta


def prepare_sequential_splits(
    *,
    n_samples: int = 200,
    seq_len: int = 8,
    input_dim: int = 2,
    noise: float = 0.1,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X, y, meta = make_sequential_binary(
        n_samples=n_samples,
        seq_len=seq_len,
        input_dim=input_dim,
        noise=noise,
        random_state=random_state,
    )
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=test_size, random_state=random_state)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    meta.update({"seq_len": seq_len, "input_dim": input_dim, "normalized": True})
    return X_train, X_test, y_train, y_test, meta
