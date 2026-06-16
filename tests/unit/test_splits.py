"""Unit tests for train/test splits."""

import numpy as np

from src.data.splits import split_train_test


def test_split_train_test_stratified(sample_binary_data):
    X, y = sample_binary_data
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert set(np.unique(y_train)) <= {0.0, 1.0}
    assert set(np.unique(y_test)) <= {0.0, 1.0}
