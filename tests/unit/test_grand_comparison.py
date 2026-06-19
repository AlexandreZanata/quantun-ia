"""Unit tests for grand comparison synthesis."""

from pathlib import Path

from src.training.grand_comparison import (
    build_grand_comparison_matrix,
    export_grand_comparison_json,
    export_grand_comparison_latex,
    grand_comparison_to_dict,
    load_grand_comparison_registry,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config" / "grand_comparison_registry.yaml"


def test_load_grand_comparison_registry():
    registry = load_grand_comparison_registry(REGISTRY)
    assert "higgs" in registry["domains"]
    assert "qnn_head_4q" in registry["recipes"]


def test_build_matrix_hypothesis_confirmed():
    registry = load_grand_comparison_registry(REGISTRY)
    result = build_grand_comparison_matrix(registry, claim_win_delta_pp=0.5)
    assert result.hypothesis_confirmed is True
    assert result.quantum_recipe_wins["qnn_head_4q"] == 0
    assert result.pending_domains.get("qnn_head_4q", []) == []


def test_export_json_and_latex(tmp_path):
    registry = load_grand_comparison_registry(REGISTRY)
    result = build_grand_comparison_matrix(registry)
    payload = grand_comparison_to_dict(result, registry)
    json_path = export_grand_comparison_json(payload, tmp_path / "nano_grand_comparison.json")
    latex_path = export_grand_comparison_latex(result, registry, tmp_path / "grand_comparison.tex")
    assert json_path.is_file()
    assert latex_path.is_file()
    assert "nano_grand_comparison" in json_path.read_text(encoding="utf-8")
    assert "\\begin{table}" in latex_path.read_text(encoding="utf-8")
