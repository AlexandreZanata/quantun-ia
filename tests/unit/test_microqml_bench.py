"""Unit tests for MicroQML Bench export builder."""

from __future__ import annotations

import json

from src.benchmark.microqml_bench import build_bench_export, load_bench_config, write_bench_export


def test_load_bench_config_has_four_primary_tasks():
    config = load_bench_config()
    primary = [t for t in config["tasks"] if t.get("primary")]
    assert len(primary) == 4
    exp_ids = {t["exp_id"] for t in primary}
    assert exp_ids == {"exp_001", "exp_011", "exp_012", "exp_014"}


def test_build_export_leaderboard_only_includes_bench_tasks():
    records = [
        {
            "exp_id": "exp_011",
            "model_name": "perceptron",
            "record_type": "multi_seed_summary",
            "started_at": "2026-06-17T10:00:00",
            "summary": {
                "perceptron": {
                    "mean": 0.91,
                    "std": 0.02,
                    "ci_low": 0.89,
                    "ci_high": 0.93,
                    "n_seeds": 10,
                }
            },
        },
        {
            "exp_id": "nano_train",
            "model_name": "perceptron",
            "test_accuracy": 0.99,
            "eval_set": "holdout_test",
            "started_at": "2026-06-17T11:00:00",
        },
    ]
    export = build_bench_export(records=records, software_version="0.9.7-test")
    assert len(export["leaderboard"]) == 1
    row = export["leaderboard"][0]
    assert row["task_id"] == "tabular_breast_cancer"
    assert row["exp_id"] == "exp_011"
    assert row["accuracy_pct"] == 91.0


def test_write_bench_export_creates_json(tmp_path):
    export = build_bench_export(records=[], software_version="0.9.7-test")
    path = write_bench_export(export, tmp_path / "bench.json")
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["bench_id"] == "microqml_bench"
