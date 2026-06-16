"""Unit tests for dataset poisoning."""

import numpy as np
import pytest

from src.data.poisoning import measure_robustness, poison_dataset


def test_poison_dataset_flips_labels():
    X = np.zeros((100, 2), dtype=np.float32)
    y = np.zeros(100, dtype=np.float32)

    _, y_poisoned, mask = poison_dataset(X, y, poison_rate=0.1, seed=42)

    assert mask.sum() == 10
    assert np.all(y_poisoned[mask] == 1.0)
    assert np.all(y_poisoned[~mask] == 0.0)


def test_measure_robustness():
    results = {0.0: 0.95, 0.1: 0.85, 0.2: 0.70}
    robustness = measure_robustness(results)

    assert robustness[0.0]["drop"] == 0.0
    assert robustness[0.1]["drop"] == pytest.approx(0.1)
    assert robustness[0.2]["drop"] == pytest.approx(0.25)
