"""Balanced subsampling and classification metrics for imbalanced holdout evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def balanced_subsample_indices(
    labels: np.ndarray,
    n_rows: int,
    *,
    min_negatives: int,
    random_state: int,
) -> np.ndarray:
    """
    Select row indices with at least ``min_negatives`` class-0 samples when available.

    Remaining slots are filled with positives. Raises when negatives are insufficient.
    """
    if n_rows <= 0:
        msg = "n_rows must be positive"
        raise ValueError(msg)
    if min_negatives < 0:
        msg = "min_negatives must be non-negative"
        raise ValueError(msg)

    labels = np.asarray(labels, dtype=int)
    neg_idx = np.flatnonzero(labels == 0)
    pos_idx = np.flatnonzero(labels == 1)

    if min_negatives > 0 and len(neg_idx) < min_negatives:
        msg = (
            f"insufficient negatives: need {min_negatives}, "
            f"have {len(neg_idx)} in split"
        )
        raise ValueError(msg)

    n_neg_target = min(min_negatives, len(neg_idx), n_rows)
    n_pos_target = n_rows - n_neg_target

    if n_pos_target > len(pos_idx):
        n_pos_target = len(pos_idx)
        n_neg_target = min(n_rows - n_pos_target, len(neg_idx))

    rng = np.random.default_rng(random_state)
    selected_neg = (
        rng.choice(neg_idx, size=n_neg_target, replace=False)
        if n_neg_target > 0
        else np.array([], dtype=int)
    )
    selected_pos = (
        rng.choice(pos_idx, size=n_pos_target, replace=False)
        if n_pos_target > 0
        else np.array([], dtype=int)
    )
    selected = np.concatenate([selected_neg, selected_pos])
    rng.shuffle(selected)
    return selected


def roc_auc(y_true: list[int] | np.ndarray, probs: list[float] | np.ndarray) -> float | None:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probs))


def pr_auc(y_true: list[int] | np.ndarray, probs: list[float] | np.ndarray) -> float | None:
    labels = np.asarray(y_true, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, probs))


def brier_score(y_true: list[int] | np.ndarray, probs: list[float] | np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(probs, dtype=float)))


def expected_calibration_error(
    y_true: list[int] | np.ndarray,
    probs: list[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error (ECE) with uniform probability bins."""
    labels = np.asarray(y_true, dtype=float)
    predictions = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    if len(labels) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        low = bin_edges[idx]
        high = bin_edges[idx + 1]
        if idx < n_bins - 1:
            mask = (predictions >= low) & (predictions < high)
        else:
            mask = (predictions >= low) & (predictions <= high)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(labels[mask]))
        bin_conf = float(np.mean(predictions[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)
    return float(ece)
