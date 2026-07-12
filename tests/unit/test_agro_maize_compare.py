"""Unit tests for agro maize compare floors loader."""

from __future__ import annotations

from src.application.agro_maize_compare import load_maize_published_floors


def test_load_maize_published_floors_includes_histgb_and_quantum():
    floors = load_maize_published_floors()
    names = " ".join(r.name for r in floors)
    assert "HistGradientBoosting" in names
    assert any("distill" in r.name.lower() for r in floors)
    assert any(r.source == "exp_086" for r in floors)
