"""Unit tests for Cycle v2 paper table export."""

from __future__ import annotations

from pathlib import Path

from src.training.cycle_v2_tables import (
    DEFAULT_REGISTRY,
    export_cycle_v2_tables,
    load_cycle_v2_registry,
)


def test_cycle_v2_registry_has_required_blocks():
    registry = load_cycle_v2_registry(DEFAULT_REGISTRY)
    for key in (
        "boosting_frontier",
        "sample_efficiency",
        "quantum_v2",
        "multicrop",
        "hard_drift",
    ):
        assert key in registry
        assert "rows" in registry[key]
        assert len(registry[key]["rows"]) >= 2
    quantum_exps = {row["experiment"] for row in registry["quantum_v2"]["rows"]}
    assert "exp_087" in quantum_exps


def test_export_cycle_v2_tables_writes_files(tmp_path: Path):
    created = export_cycle_v2_tables(out_dir=tmp_path)
    names = {p.name for p in created}
    assert names == {
        "boosting_frontier.tex",
        "sample_efficiency.tex",
        "quantum_v2.tex",
        "multicrop.tex",
        "hard_drift.tex",
    }
    text = (tmp_path / "boosting_frontier.tex").read_text(encoding="utf-8")
    assert "0.8130" in text
    assert "tab:boosting_frontier" in text
    assert "HistGradientBoosting" in text
    quantum = (tmp_path / "quantum_v2.tex").read_text(encoding="utf-8")
    assert "exp_087" in quantum
    assert "Fourier" in quantum
    drift = (tmp_path / "hard_drift.tex").read_text(encoding="utf-8")
    assert "0.8185" in drift
    assert "tab:hard_drift" in drift
