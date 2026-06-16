"""Unit tests for quantum amplitude encoding."""

import numpy as np

from src.quantum.amplitude_encoding import (
    angle_encode,
    normalize_for_amplitude,
    pad_to_power_of_two,
)


def test_normalize_for_amplitude():
    X = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = normalize_for_amplitude(X)

    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    assert np.all(normed[1] == 0.0)


def test_pad_to_power_of_two():
    X = np.ones((5, 3), dtype=np.float32)
    padded = pad_to_power_of_two(X)

    assert padded.shape == (5, 4)


def test_angle_encode():
    X = np.array([[0.5, -0.5]], dtype=np.float32)
    encoded = angle_encode(X)

    assert encoded.max() <= np.pi
    assert encoded.min() >= -np.pi
