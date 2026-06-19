"""Contract tests for primary paper narrative (Option C, Phase 25)."""

from __future__ import annotations

from pathlib import Path

INTRO = Path("paper/sections/introduction.tex")
RESULTS = Path("paper/sections/results.tex")

HEADLINE_MARKERS = ("exp_011", "Option", "exp_021", "exp_022")
DEFERRED_MARKERS = ("exp_015", "exp_016", "adaptive LR", "NAS")


def test_introduction_states_option_c_narrative():
    text = INTRO.read_text(encoding="utf-8")
    assert "Option" in text and "honest" in text.lower()
    normalized = text.replace("\\", "")
    assert "exp_011" in normalized


def test_introduction_defers_innovation_tracks():
    text = INTRO.read_text(encoding="utf-8")
    assert "deferred" in text.lower() or "follow-up" in text.lower()
    for marker in ("exp_015", "exp_016"):
        assert marker not in text, f"introduction must not cite {marker} as headline"


def test_results_cover_headline_experiments():
    text = RESULTS.read_text(encoding="utf-8")
    normalized = text.replace("\\", "")
    for marker in ("exp_021", "exp_022", "exp_011", "exp_068"):
        assert marker in normalized, f"results missing headline marker: {marker}"


def test_results_do_not_claim_deferred_methods():
    text = RESULTS.read_text(encoding="utf-8")
    for marker in DEFERRED_MARKERS:
        assert marker not in text, f"results must not cite deferred track: {marker}"
