"""Unit tests for publication figure generation."""

import json

import matplotlib

matplotlib.use("Agg")

from dashboard.benchmark_data import to_leaderboard_rows
from scripts.generate_figures import (
    generate_all_figures,
    plot_cross_experiment_best,
    plot_experiment_leaderboard,
    plot_learning_curves,
)


def _sample_records() -> list[dict]:
    return [
        {
            "exp_id": "exp_001",
            "record_type": "multi_seed_summary",
            "started_at": "2026-06-16T12:00:00",
            "summary": {
                "classical_32": {
                    "mean": 0.65,
                    "std": 0.04,
                    "ci_low": 0.62,
                    "ci_high": 0.68,
                    "n_seeds": 10,
                },
                "quantum_basic": {
                    "mean": 0.60,
                    "std": 0.05,
                    "ci_low": 0.57,
                    "ci_high": 0.63,
                    "n_seeds": 10,
                },
            },
        },
        {
            "exp_id": "exp_001",
            "model_name": "classical_32_seed42",
            "test_accuracy": 0.65,
            "eval_set": "holdout_test",
            "started_at": "2026-06-16T10:00:00",
            "history": [
                {"epoch": 1, "accuracy": 0.5},
                {"epoch": 2, "accuracy": 0.6},
            ],
        },
    ]


def test_plot_experiment_leaderboard_creates_pdf(tmp_path):
    rows = to_leaderboard_rows(_sample_records())
    out = tmp_path / "exp_001_leaderboard.pdf"
    assert plot_experiment_leaderboard("exp_001", rows, out) is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_cross_experiment_best_creates_pdf(tmp_path):
    rows = to_leaderboard_rows(_sample_records())
    out = tmp_path / "cross.pdf"
    assert plot_cross_experiment_best(rows, out) is True
    assert out.exists()


def test_plot_learning_curves_creates_pdf(tmp_path):
    out = tmp_path / "curves.pdf"
    assert plot_learning_curves(_sample_records(), out) is True
    assert out.exists()


def test_generate_all_figures_from_jsonl(tmp_path):
    jsonl = tmp_path / "experiments.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in _sample_records()))
    out_dir = tmp_path / "figures"
    created = generate_all_figures(jsonl_path=jsonl, out_dir=out_dir)
    assert len(created) >= 2
    assert all(p.exists() for p in created)
