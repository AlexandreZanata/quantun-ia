"""Knowledge-distillation helpers for conventional → nano soft-label training."""

from __future__ import annotations

import numpy as np


def soft_targets_from_proba(proba_2d: np.ndarray) -> np.ndarray:
    """Extract positive-class probabilities from predict_proba output."""
    arr = np.asarray(proba_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected predict_proba array with shape (n, 2+)")
    return arr[:, 1].astype(np.float64, copy=False)


def mix_hard_soft_targets(
    hard: np.ndarray,
    soft: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Mix soft teacher probs with hard labels: ``alpha * soft + (1 - alpha) * hard``.

    ``alpha=1`` is pure distillation; ``alpha=0`` is hard-label training.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    h = np.asarray(hard, dtype=np.float64).reshape(-1)
    s = np.asarray(soft, dtype=np.float64).reshape(-1)
    if h.shape != s.shape:
        raise ValueError(f"hard/soft shape mismatch: {h.shape} vs {s.shape}")
    mixed = alpha * s + (1.0 - alpha) * h
    return np.clip(mixed, 0.0, 1.0)
