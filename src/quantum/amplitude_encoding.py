"""Quantum data encoding — amplitude and angle encoding."""

import numpy as np


def normalize_for_amplitude(X: np.ndarray) -> np.ndarray:
    """Normalize each vector to unit norm (required for amplitude encoding)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def pad_to_power_of_two(X: np.ndarray) -> np.ndarray:
    """Pad features to the next power of two (required for amplitude encoding)."""
    n_features = X.shape[1]
    target = 1
    while target < n_features:
        target *= 2
    if target == n_features:
        return X
    pad_width = target - n_features
    return np.pad(X, ((0, 0), (0, pad_width)), mode="constant")


def angle_encode(X: np.ndarray, scale: float = np.pi) -> np.ndarray:
    """Scale features to [-scale, scale] for AngleEmbedding."""
    return np.clip(X * scale, -scale, scale)
