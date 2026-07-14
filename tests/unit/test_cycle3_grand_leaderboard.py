"""Unit tests for Cycle v3 grand leaderboard synthesis."""

from pathlib import Path

from src.training.cycle3_grand_leaderboard import (
    REQUIRED_EXPERIMENTS,
    build_cycle3_grand_leaderboard,
    cycle3_leaderboard_to_dict,
    export_cycle3_grand_leaderboard_json,
    export_cycle3_grand_leaderboard_latex,
    load_cycle3_grand_leaderboard_registry,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config" / "cycle3_grand_leaderboard_registry.yaml"


def test_load_cycle3_grand_leaderboard_registry():
    registry = load_cycle3_grand_leaderboard_registry(REGISTRY)
    assert "rows" in registry
    assert len(registry["rows"]) == len(REQUIRED_EXPERIMENTS)


def test_build_cycle3_grand_leaderboard_confirms_hypothesis():
    registry = load_cycle3_grand_leaderboard_registry(REGISTRY)
    result = build_cycle3_grand_leaderboard(registry)
    assert result.coverage_ok
    assert result.quantum_honesty_ok
    assert result.accepts_ok
    assert result.hypothesis_confirmed
    assert result.observed_accepts == frozenset(
        {"exp_101", "exp_102", "exp_106", "exp_109"}
    )
    assert result.quantum_false_claims == ()


def test_export_cycle3_grand_leaderboard_artifacts(tmp_path: Path):
    registry = load_cycle3_grand_leaderboard_registry(REGISTRY)
    result = build_cycle3_grand_leaderboard(registry)
    payload = cycle3_leaderboard_to_dict(result, registry)
    json_path = export_cycle3_grand_leaderboard_json(
        payload, tmp_path / "cycle3_grand_leaderboard.json"
    )
    latex_path = export_cycle3_grand_leaderboard_latex(
        result, tmp_path / "cycle3_grand_leaderboard.tex"
    )
    assert "cycle3_grand_leaderboard" in json_path.read_text(encoding="utf-8")
    text = latex_path.read_text(encoding="utf-8")
    assert "tab:cycle3_grand_leaderboard" in text
    assert "exp_102" in text or "exp\\_102" in text
