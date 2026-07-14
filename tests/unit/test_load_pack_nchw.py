"""Unit: Phase L pack NCHW loader shapes (skips if packs missing)."""

from __future__ import annotations

import pytest

from src.data.open_images import is_pack_complete, load_pack_nchw


@pytest.mark.skipif(not is_pack_complete("tiny_imagenet"), reason="tiny_imagenet not downloaded")
def test_load_tiny_imagenet_nchw_shape():
    x, y = load_pack_nchw("tiny_imagenet", split="train", n_take=4, img_size=64, seed=0)
    assert x.shape == (4, 3, 64, 64)
    assert y.shape == (4,)
    assert float(x.min()) >= -1.01
    assert float(x.max()) <= 1.01


@pytest.mark.skipif(not is_pack_complete("stl10"), reason="stl10 not downloaded")
def test_load_stl10_nchw_resize_64():
    x, y = load_pack_nchw("stl10", split="train", n_take=4, img_size=64, seed=0)
    assert x.shape == (4, 3, 64, 64)
    assert y.shape == (4,)
