"""Unit smoke — open image download script helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.download_open_images import PACKS, write_generation_md, write_stats


def test_packs_are_p0_triple():
    assert PACKS == ("cifar10", "fashion_mnist", "flowers102")


def test_generation_md_written(tmp_path: Path):
    results = [{"pack": "cifar10", "path": str(tmp_path / "cifar10"), "skipped": True}]
    path = write_generation_md(tmp_path, results)
    text = path.read_text(encoding="utf-8")
    assert "cifar10" in text
    assert "License" in text


def test_download_stats_written(tmp_path: Path):
    results = [{"pack": "fashion_mnist", "path": "/tmp/x", "skipped": False}]
    path = write_stats(tmp_path, results)
    assert path.is_file()
    assert "fashion_mnist" in path.read_text(encoding="utf-8")
