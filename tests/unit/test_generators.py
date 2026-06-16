"""Unit tests for synthetic data generators."""

import numpy as np

from src.data.generators import make_binary_classification


def test_make_binary_classification_moons():
    X, y, scaler = make_binary_classification(n_samples=100, dataset="moons")

    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert X.dtype == np.float32
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_make_binary_classification_reproducible():
    X1, y1, _ = make_binary_classification(random_state=7)
    X2, y2, _ = make_binary_classification(random_state=7)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
