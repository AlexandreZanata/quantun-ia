"""Unit tests for paper build helpers."""

from pathlib import Path

from scripts.build_paper import sync_paper_assets


def test_sync_paper_assets_copies_pdfs(tmp_path: Path):
    src = tmp_path / "figures"
    dst = tmp_path / "paper" / "figures"
    src.mkdir()
    sample = src / "exp_001_leaderboard.pdf"
    sample.write_bytes(b"%PDF-1.4")

    copied = sync_paper_assets(figures_src=src, figures_dst=dst)

    assert len(copied) == 1
    assert (dst / "exp_001_leaderboard.pdf").read_bytes() == b"%PDF-1.4"
