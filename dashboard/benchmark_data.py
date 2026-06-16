"""Load and normalize benchmark records from logs/experiments.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.param_match import build_param_match_table

LOGS_PATH = Path(__file__).resolve().parents[1] / "logs" / "experiments.jsonl"
SKIP_RECORD_TYPES = frozenset({"multi_seed_summary", "paired_comparison", "applicability_gate"})


def load_records() -> list[dict]:
    records = []
    if LOGS_PATH.exists():
        with open(LOGS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def to_benchmark_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        if r.get("record_type") in SKIP_RECORD_TYPES:
            continue

        acc = r.get("test_accuracy")
        if acc is None:
            acc = r.get("final_acc")
        loss = r.get("test_loss")
        if loss is None:
            loss = r.get("final_loss")
        eval_set = r.get("eval_set", "train" if r.get("test_accuracy") is None else "holdout_test")
        rows.append(
            {
                "exp_id": r.get("exp_id", "?"),
                "model": r.get("model_name", "?"),
                "accuracy": round(acc * 100, 1) if acc is not None else None,
                "loss": round(loss, 4) if loss is not None else None,
                "eval_set": eval_set,
                "n_params": r.get("n_params"),
                "elapsed_s": round(r.get("elapsed_s", 0), 2),
                "epochs": r.get("n_epochs", 0),
                "started_at": (r.get("started_at") or "")[:16],
                "history": r.get("history", []),
            }
        )
    return rows


def load_applicability_gates(records: list[dict]) -> list[dict]:
    """Return technique applicability records for dashboard N/A display."""
    gates = []
    for r in records:
        if r.get("record_type") != "applicability_gate":
            continue
        gates.append(
            {
                "exp_id": r.get("exp_id", "?"),
                "technique": r.get("technique", "?"),
                "status": r.get("status", "?"),
                "applicable": r.get("applicable", False),
                "mean_holdout_pct": round(r.get("mean_holdout", 0) * 100, 1),
                "threshold_pct": round(r.get("threshold", 0.55) * 100, 1),
                "reason": r.get("reason", ""),
            }
        )
    return gates


def param_match_table(records: list[dict]) -> list[dict]:
    """Delegate to training.param_match for parameter-matched baseline table."""
    return build_param_match_table(records)


def best_row(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r["accuracy"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: r["accuracy"])
