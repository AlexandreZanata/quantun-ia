"""Data augmentation utilities for experiments."""

import numpy as np


def add_gaussian_noise(X, sigma: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    return X + rng.normal(0, sigma, X.shape).astype(X.dtype)


def random_flip_labels(y, flip_rate: float = 0.1, seed: int = 42):
    """Light augmentation — partial label flip (distinct from poisoning)."""
    rng = np.random.default_rng(seed)
    y_aug = y.copy()
    n_flip = int(len(y) * flip_rate)
    idx = rng.choice(len(y), size=n_flip, replace=False)
    y_aug[idx] = 1 - y_aug[idx]
    return y_aug
