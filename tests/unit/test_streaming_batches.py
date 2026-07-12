"""Unit tests for chronological streaming batches."""

from __future__ import annotations

import numpy as np

from src.training.streaming_batches import chronological_batches


def test_chronological_batches_contiguous():
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    y = np.arange(10, dtype=np.float64)
    batches = chronological_batches(x, y, n_batches=5)
    assert len(batches) == 5
    assert [len(b[1]) for b in batches] == [2, 2, 2, 2, 2]
    # Order preserved
    assert list(batches[0][1]) == [0.0, 1.0]
    assert list(batches[-1][1]) == [8.0, 9.0]


def test_chronological_batches_remainder():
    x = np.zeros((11, 1), dtype=np.float64)
    y = np.arange(11, dtype=np.float64)
    batches = chronological_batches(x, y, n_batches=3)
    assert sum(len(b[1]) for b in batches) == 11
    assert len(batches[-1][1]) == 11 - 2 * (11 // 3)
