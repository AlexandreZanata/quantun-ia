"""Curriculum learning — order examples by difficulty."""

import numpy as np


def sort_by_difficulty(X, y, method="margin"):
    """
    method='margin': distance to class centroid
    method='random': random baseline
    """
    if method == "random":
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)

    ease_scores = np.array([
        np.linalg.norm(X[i] - (c0 if y[i] == 0 else c1)) * -1
        for i in range(len(X))
    ])
    idx = np.argsort(ease_scores)
    return X[idx], y[idx]


def curriculum_batches(X, y, n_stages: int = 4):
    """Split the dataset into stages of increasing difficulty."""
    X_sorted, y_sorted = sort_by_difficulty(X, y, method="margin")
    stage_size = len(X) // n_stages
    batches = []
    for stage in range(1, n_stages + 1):
        end = stage * stage_size if stage < n_stages else len(X)
        batches.append((X_sorted[:end], y_sorted[:end]))
    return batches
