"""Helpers for sample-efficiency learning curves (row-budget subsampling)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_row_budget(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fraction: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified subset with approximately ``fraction`` of rows.

    ``fraction`` in (0, 1]; ``1.0`` returns the full arrays unchanged.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return x, y
    n = len(y)
    train_size = max(int(round(n * fraction)), 2)
    # Ensure both classes if possible
    if len(np.unique(y)) >= 2:
        idx, _ = train_test_split(
            np.arange(n),
            train_size=min(train_size, n - 1),
            stratify=y,
            random_state=random_state,
        )
    else:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=min(train_size, n), replace=False)
    return x[idx], y[idx]


def area_under_budget_curve(fractions: list[float], aucs: list[float]) -> float:
    """Trapezoidal AUC over log-ish or linear fraction axis (linear in fraction)."""
    if len(fractions) != len(aucs) or len(fractions) < 2:
        raise ValueError("need ≥2 matched fraction/auc points")
    order = np.argsort(np.asarray(fractions, dtype=np.float64))
    f = np.asarray(fractions, dtype=np.float64)[order]
    a = np.asarray(aucs, dtype=np.float64)[order]
    return float(np.trapezoid(a, f))
