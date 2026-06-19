"""Unit tests for balanced evaluation metrics and subsampling."""

from __future__ import annotations

import numpy as np
import pytest

from src.application.balanced_metrics import (
    balanced_subsample_indices,
    brier_score,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)


def test_balanced_subsample_enforces_min_negatives():
    labels = np.array([0] * 50 + [1] * 9950)
    indices = balanced_subsample_indices(labels, n_rows=100, min_negatives=10, random_state=42)
    selected = labels[indices]
    assert len(selected) == 100
    assert int(np.sum(selected == 0)) >= 10
    assert int(np.sum(selected == 1)) == 90


def test_balanced_subsample_reproducible():
    labels = np.array([0] * 20 + [1] * 980)
    a = balanced_subsample_indices(labels, n_rows=50, min_negatives=5, random_state=7)
    b = balanced_subsample_indices(labels, n_rows=50, min_negatives=5, random_state=7)
    assert np.array_equal(a, b)


def test_balanced_subsample_raises_when_insufficient_negatives():
    labels = np.array([1] * 100)
    with pytest.raises(ValueError, match="negatives"):
        balanced_subsample_indices(labels, n_rows=50, min_negatives=5, random_state=42)


def test_roc_auc_requires_both_classes():
    assert roc_auc([1, 1, 1], [0.9, 0.8, 0.7]) is None
    score = roc_auc([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.9])
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_pr_auc_on_imbalanced():
    y_true = [0] * 90 + [1] * 10
    probs = [0.1] * 90 + [0.9] * 10
    score = pr_auc(y_true, probs)
    assert score is not None
    assert score > 0.5


def test_expected_calibration_error_perfect_is_low():
    y_true = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]
    ece = expected_calibration_error(y_true, probs, n_bins=2)
    assert 0.0 <= ece <= 0.25


def test_brier_score_bounds():
    score = brier_score([0, 1, 1], [0.1, 0.9, 0.6])
    assert 0.0 <= score <= 1.0
