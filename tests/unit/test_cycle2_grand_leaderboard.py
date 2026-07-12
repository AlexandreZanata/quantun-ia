"""Unit tests for Cycle v2 grand leaderboard synthesis."""

from pathlib import Path

from src.training.cycle2_grand_leaderboard import (
    REQUIRED_EXPERIMENTS,
    build_cycle2_grand_leaderboard,
    cycle2_leaderboard_to_dict,
    export_cycle2_grand_leaderboard_json,
    export_cycle2_grand_leaderboard_latex,
    load_cycle2_grand_leaderboard_registry,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config" / "cycle2_grand_leaderboard_registry.yaml"


def test_load_cycle2_grand_leaderboard_registry():
    registry = load_cycle2_grand_leaderboard_registry(REGISTRY)
    assert "rows" in registry
    assert len(registry["rows"]) == len(REQUIRED_EXPERIMENTS)


def test_build_cycle2_grand_leaderboard_confirms_hypothesis():
    registry = load_cycle2_grand_leaderboard_registry(REGISTRY)
    result = build_cycle2_grand_leaderboard(registry, claim_win_delta_pp=0.5)
    assert result.coverage_ok
    assert result.quantum_honesty_ok
    assert result.accepts_ok
    assert result.hypothesis_confirmed
    assert result.observed_accepts == frozenset(
        {"exp_091", "exp_092", "exp_094", "exp_096", "exp_097"}
    )
    assert result.quantum_claim_wins == ()


def test_export_cycle2_grand_leaderboard_artifacts(tmp_path: Path):
    registry = load_cycle2_grand_leaderboard_registry(REGISTRY)
    result = build_cycle2_grand_leaderboard(registry)
    payload = cycle2_leaderboard_to_dict(result, registry)
    json_path = export_cycle2_grand_leaderboard_json(
        payload, tmp_path / "cycle2_grand_leaderboard.json"
    )
    latex_path = export_cycle2_grand_leaderboard_latex(
        result, tmp_path / "cycle2_grand_leaderboard.tex"
    )
    assert "cycle2_grand_leaderboard" in json_path.read_text(encoding="utf-8")
    assert "tab:cycle2_grand_leaderboard" in latex_path.read_text(encoding="utf-8")
    assert "exp\\_092" in latex_path.read_text(encoding="utf-8") or "exp_092" in latex_path.read_text(
        encoding="utf-8"
    )
