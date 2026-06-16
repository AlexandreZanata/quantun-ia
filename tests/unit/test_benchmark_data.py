"""Unit tests for dashboard benchmark row normalization."""

from dashboard.benchmark_data import (
    is_holdout_record,
    latest_holdout_records,
    load_applicability_gates,
    param_match_table,
    to_benchmark_rows,
    to_leaderboard_rows,
)


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


def test_load_applicability_gates():
    records = [
        {
            "exp_id": "exp_005",
            "record_type": "applicability_gate",
            "technique": "curriculum",
            "status": "not_applicable",
            "applicable": False,
            "mean_holdout": 0.52,
            "threshold": 0.55,
            "reason": "below threshold",
        }
    ]
    gates = load_applicability_gates(records)
    assert gates[0]["status"] == "not_applicable"
    assert gates[0]["mean_holdout_pct"] == 52.0


def test_param_match_table_delegates():
    records = [
        {
            "exp_id": "exp_008",
            "model_name": "quantum_reupload_seed42",
            "n_params": 30,
            "test_accuracy": 0.6,
        }
    ]
    rows = param_match_table(records)
    assert rows[0]["model"] == "quantum_reupload"


def test_is_holdout_record_excludes_train_only():
    assert is_holdout_record({"test_accuracy": 0.7, "eval_set": "holdout_test"}) is True
    assert is_holdout_record({"final_acc": 0.9, "eval_set": "train"}) is False
    assert is_holdout_record({"record_type": "multi_seed_summary"}) is False


def test_latest_holdout_records_keeps_newest_per_model():
    records = [
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "test_accuracy": 0.60,
            "eval_set": "holdout_test",
            "started_at": "2026-06-16T09:00:00",
        },
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "test_accuracy": 0.65,
            "eval_set": "holdout_test",
            "started_at": "2026-06-16T10:00:00",
        },
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "final_acc": 0.87,
            "eval_set": "train",
            "started_at": "2026-06-16T11:00:00",
        },
    ]
    latest = latest_holdout_records(records)
    assert len(latest) == 1
    assert latest[0]["test_accuracy"] == 0.65


def test_to_leaderboard_rows_dedupes_seed_suffix():
    records = [
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "test_accuracy": 0.65,
            "eval_set": "holdout_test",
            "started_at": "2026-06-16T10:00:00",
            "elapsed_s": 1.0,
            "n_epochs": 50,
        },
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed123",
            "test_accuracy": 0.64,
            "eval_set": "holdout_test",
            "started_at": "2026-06-16T09:00:00",
            "elapsed_s": 1.0,
            "n_epochs": 50,
        },
    ]
    rows = to_leaderboard_rows(records)
    assert len(rows) == 1
    assert rows[0]["model"] == "classical_32"
    assert rows[0]["accuracy"] == 65.0


def test_load_applicability_gates_keeps_latest():
    records = [
        {
            "exp_id": "exp_005",
            "record_type": "applicability_gate",
            "technique": "curriculum",
            "status": "not_applicable",
            "applicable": False,
            "mean_holdout": 0.52,
            "threshold": 0.55,
            "reason": "old",
            "started_at": "2026-06-16T09:00:00",
        },
        {
            "exp_id": "exp_005",
            "record_type": "applicability_gate",
            "technique": "curriculum",
            "status": "applicable",
            "applicable": True,
            "mean_holdout": 0.60,
            "threshold": 0.55,
            "reason": "new",
            "started_at": "2026-06-16T10:00:00",
        },
    ]
    gates = load_applicability_gates(records)
    assert len(gates) == 1
    assert gates[0]["status"] == "applicable"
