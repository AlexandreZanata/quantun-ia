"""Unit tests for JSONL → CSV export."""

import json

from scripts.export_results import export_jsonl_to_csv


def test_export_jsonl_to_csv_writes_summary_rows(tmp_path):
    jsonl = tmp_path / "experiments.jsonl"
    csv_path = tmp_path / "results.csv"
    record = {
        "exp_id": "exp_001",
        "model_name": "classical_8_seed42",
        "seed": 42,
        "profile": "ci",
        "started_at": "2026-01-01T00:00:00",
        "elapsed_s": 1.2,
        "final_acc": 0.8,
        "test_accuracy": 0.75,
        "eval_set": "holdout_test",
        "n_epochs": 5,
        "n_params": 100,
    }
    jsonl.write_text(json.dumps(record) + "\n")

    count = export_jsonl_to_csv(jsonl_path=jsonl, csv_path=csv_path)
    assert count == 1
    text = csv_path.read_text()
    assert "exp_001" in text
    assert "classical_8_seed42" in text
