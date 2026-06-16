"""Unit tests for curriculum learning."""

import numpy as np

from src.training.curriculum import curriculum_batches, sort_by_difficulty


def test_sort_by_difficulty_random(sample_binary_data):
    X, y = sample_binary_data
    X_sorted, y_sorted = sort_by_difficulty(X, y, method="random")

    assert X_sorted.shape == X.shape
    assert y_sorted.shape == y.shape


def test_sort_by_difficulty_margin(sample_binary_data):
    X, y = sample_binary_data
    X_sorted, y_sorted = sort_by_difficulty(X, y, method="margin")

    assert X_sorted.shape == X.shape
    assert set(np.unique(y_sorted)) <= {0.0, 1.0}


def test_curriculum_batches(sample_binary_data):
    X, y = sample_binary_data
    batches = curriculum_batches(X, y, n_stages=4)

    assert len(batches) == 4
    assert batches[0][0].shape[0] <= batches[-1][0].shape[0]
