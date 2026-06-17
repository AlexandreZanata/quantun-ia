"""Unit tests for LaTeX table export."""

import json

from scripts.export_latex_tables import export_latex_tables, summary_to_latex


def _sample_jsonl(tmp_path):
    jsonl = tmp_path / "experiments.jsonl"
    record = {
        "exp_id": "exp_001",
        "record_type": "multi_seed_summary",
        "started_at": "2026-06-16T12:00:00",
        "summary": {
            "classical_32": {
                "mean": 0.65,
                "ci_low": 0.62,
                "ci_high": 0.68,
                "n_seeds": 10,
            }
        },
    }
    jsonl.write_text(json.dumps(record) + "\n")
    return jsonl


def test_summary_to_latex_contains_model_and_ci():
    tex = summary_to_latex(
        "exp_001",
        {"classical_32": {"mean": 0.65, "ci_low": 0.62, "ci_high": 0.68}},
    )
    assert "classical_32" in tex
    assert "65.0" in tex
    assert "\\begin{table}" in tex
    assert "\\toprule" in tex


def test_export_latex_tables_writes_files(tmp_path):
    jsonl = _sample_jsonl(tmp_path)
    out_dir = tmp_path / "tables"
    created = export_latex_tables(jsonl_path=jsonl, out_dir=out_dir)
    assert len(created) >= 2
    holdout = out_dir / "exp_001_holdout.tex"
    assert holdout.exists()
    assert "classical_32" in holdout.read_text()
    combined = (out_dir / "all_experiments_summary.tex").read_text()
    assert "\\input{tables/exp_001_holdout}" in combined
