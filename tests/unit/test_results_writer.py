"""Unit tests for results.md generation from JSONL."""

import json

import pytest

from src.training.results_writer import generate_results_md


def test_generate_results_md_includes_summary_table(tmp_path):
    jsonl = tmp_path / "experiments.jsonl"
    records = [
        {
            "exp_id": "exp_011",
            "record_type": "multi_seed_summary",
            "started_at": "2026-06-17T12:00:00",
            "summary": {
                "perceptron": {
                    "mean": 0.92,
                    "ci_low": 0.90,
                    "ci_high": 0.94,
                    "n_seeds": 10,
                },
                "quantum_angle": {
                    "mean": 0.88,
                    "ci_low": 0.85,
                    "ci_high": 0.91,
                    "n_seeds": 10,
                },
            },
        },
        {
            "exp_id": "exp_011",
            "model_name": "perceptron_seed42",
            "profile": "publication",
            "seed": 42,
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    md = generate_results_md(
        "exp_011",
        "EXP 011 Test",
        jsonl_path=jsonl,
        dataset_note="breast_cancer",
    )
    assert "perceptron" in md
    assert "92.0%" in md
    assert "## Limitations" in md


def test_generate_results_md_raises_without_summary(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text('{"exp_id": "exp_011", "model_name": "x"}\n')
    with pytest.raises(ValueError, match="multi_seed_summary"):
        generate_results_md("exp_011", "EXP 011", jsonl_path=jsonl)
