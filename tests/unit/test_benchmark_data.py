"""Unit tests for dashboard benchmark row normalization."""

from dashboard.benchmark_data import to_benchmark_rows


def test_to_benchmark_rows_includes_eval_set():
    records = [
        {
            "exp_id": "exp_test",
            "model_name": "model_a",
            "final_acc": 0.8,
            "test_accuracy": 0.75,
            "eval_set": "holdout_test",
            "elapsed_s": 1.0,
            "n_epochs": 10,
            "started_at": "2026-06-16T10:00:00",
        }
    ]
    rows = to_benchmark_rows(records)
    assert rows[0]["eval_set"] == "holdout_test"
    assert rows[0]["accuracy"] == 75.0


def test_to_benchmark_rows_skips_research_summaries():
    records = [
        {
            "exp_id": "exp_001",
            "model_name": "exp_001_multi_seed_summary",
            "record_type": "multi_seed_summary",
            "summary": {},
        },
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "test_accuracy": 0.85,
            "eval_set": "holdout_test",
            "elapsed_s": 1.0,
            "n_epochs": 50,
        },
    ]
    rows = to_benchmark_rows(records)
    assert len(rows) == 1
    assert rows[0]["model"] == "classical_32_seed42"
