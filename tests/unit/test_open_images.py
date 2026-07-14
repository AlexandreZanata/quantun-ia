"""Unit tests for open image pack loader helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.open_images import (
    PACK_LOADERS,
    is_pack_complete,
    load_image_pack_arrays,
    summarize_packs,
)


def test_pack_loaders_are_p0_triple():
    assert PACK_LOADERS == ("cifar10", "fashion_mnist", "flowers102")


def test_summarize_packs_keys():
    rows = summarize_packs()
    assert {r["pack"] for r in rows} == set(PACK_LOADERS)
    assert all("complete" in r and "raw_dir" in r for r in rows)


@pytest.mark.skipif(
    not is_pack_complete("fashion_mnist"),
    reason="Fashion-MNIST not downloaded (run make data-open-images-smoke)",
)
def test_load_fashion_mnist_smoke_shapes():
    batch = load_image_pack_arrays("fashion_mnist", n_train=8, n_test=8)
    assert batch["x_train"].shape == (8, 28, 28)
    assert batch["y_train"].shape == (8,)
    assert batch["x_test"].shape == (8, 28, 28)
    assert batch["spatial_shape"] == (28, 28)


def test_load_missing_pack_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_image_pack_arrays("cifar10", root=tmp_path, n_train=2, n_test=2)
