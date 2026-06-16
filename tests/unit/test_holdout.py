"""Unit tests for holdout research summaries."""

import json

from src.training.holdout import compare_conditions, summarize_multi_seed


def test_summarize_multi_seed_writes_jsonl(temp_log_file):
    summary = summarize_multi_seed(
        "exp_test",
        {"model_a": [0.8, 0.85, 0.9], "model_b": [0.75, 0.8, 0.82]},
    )

    assert summary["model_a"]["mean"] == 0.85
    assert summary["model_a"]["ci_low"] <= summary["model_a"]["mean"]

    lines = temp_log_file.read_text().strip().split("\n")
    record = json.loads(lines[-1])
    assert record["record_type"] == "multi_seed_summary"
    assert "model_a" in record["summary"]


def test_compare_conditions_writes_paired_record(temp_log_file):
    compare_conditions(
        "exp_test",
        [0.85, 0.88, 0.90],
        [0.80, 0.82, 0.84],
        "condition_a",
        "condition_b",
    )

    lines = temp_log_file.read_text().strip().split("\n")
    record = json.loads(lines[-1])
    assert record["record_type"] == "paired_comparison"
    assert record["comparison"]["mean_diff"] > 0
