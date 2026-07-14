"""CIFAR split loader respects Phase G indices."""

from __future__ import annotations

import pytest

from src.data.open_images import is_pack_complete, load_cifar10_nchw


@pytest.mark.skipif(not is_pack_complete("cifar10"), reason="cifar10 pack not on disk")
def test_load_cifar10_nchw_shapes():
    x, y = load_cifar10_nchw(split="train", n_take=8, seed=0)
    assert x.shape == (8, 3, 32, 32)
    assert y.shape == (8,)
    assert x.min() >= -1.0 - 1e-5
    assert x.max() <= 1.0 + 1e-5
