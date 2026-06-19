"""Unit tests for ACYD seasonal cyclic feature extraction."""

import torch

from src.data.acyd_cyclic import N_CYCLIC_FEATURES, extract_acyd_seasonal_cyclic


def test_extract_acyd_seasonal_cyclic_shape():
    x = torch.randn(8, 37)
    cyclic = extract_acyd_seasonal_cyclic(x)
    assert cyclic.shape == (8, N_CYCLIC_FEATURES)


def test_extract_acyd_seasonal_cyclic_bounded():
    x = torch.zeros(4, 37)
    cyclic = extract_acyd_seasonal_cyclic(x)
    assert float(cyclic.abs().max()) <= 3.15
