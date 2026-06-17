"""Unit tests for sequential_phase dataset."""

import numpy as np

from src.data.real_datasets import make_sequential_phase


def test_make_sequential_phase_shapes():
    X, y, meta = make_sequential_phase(n_samples=50, seq_len=12, input_dim=4, random_state=42)
    assert X.shape == (50, 12, 4)
    assert y.shape == (50,)
    assert meta["name"] == "sequential_phase"
    assert set(np.unique(y)).issubset({0.0, 1.0})
