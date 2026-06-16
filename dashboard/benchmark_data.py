"""Load and normalize benchmark records from logs/experiments.jsonl."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.training.param_match import build_param_match_table

LOGS_PATH = Path(__file__).resolve().parents[1] / "logs" / "experiments.jsonl"
SKIP_RECORD_TYPES = frozenset({
    "multi_seed_summary",
    "paired_comparison",
    "paired_comparison_batch",
    "applicability_gate",
})
_SEED_SUFFIX = re.compile(r"_seed\d+$")
_PUBLICATION_EXP = re.compile(r"^exp_\d{3}$")


def _base_model_name(model_name: str) -> str:
    return _SEED_SUFFIX.sub("", model_name)


def base_model_label(model_name: str) -> str:
    """Strip per-seed suffix for leaderboard display."""
    return _base_model_name(model_name)


def is_publication_experiment(exp_id: str) -> bool:
    """Only numbered lab experiments belong on the publication leaderboard."""
    return _PUBLICATION_EXP.match(exp_id) is not None


def load_records() -> list[dict]:
    records = []
    if LOGS_PATH.exists():
        with open(LOGS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def is_holdout_record(record: dict) -> bool:
    """True for per-run records with holdout test metrics (excludes train-only logs)."""
    if record.get("record_type") in SKIP_RECORD_TYPES:
        return False
    if record.get("test_accuracy") is None:
        return False
    return record.get("eval_set", "holdout_test") == "holdout_test"


def latest_holdout_records(records: list[dict]) -> list[dict]:
    """Keep the latest holdout record per (exp_id, model base name)."""
    latest: dict[tuple[str, str], dict] = {}
    for record in records:
        if not is_holdout_record(record):
            continue
        key = (record.get("exp_id", "?"), _base_model_name(record.get("model_name", "?")))
        prev = latest.get(key)
        if prev is None or (record.get("started_at") or "") >= (prev.get("started_at") or ""):
            latest[key] = record
    return list(latest.values())


def latest_multi_seed_summaries(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Latest multi-seed summary per exp_id: {exp_id: {model: stats}}."""
    latest: dict[str, dict[str, dict]] = {}
    latest_at: dict[str, str] = {}
    for record in records:
        if record.get("record_type") != "multi_seed_summary":
            continue
        exp_id = record.get("exp_id", "?")
        started = record.get("started_at") or ""
        if exp_id not in latest_at or started >= latest_at[exp_id]:
            latest_at[exp_id] = started
            latest[exp_id] = record.get("summary", {})
    return latest


def _holdout_meta_by_model(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Auxiliary training metadata from latest per-seed holdout runs."""
    meta: dict[tuple[str, str], dict] = {}
    for record in latest_holdout_records(records):
        key = (record.get("exp_id", "?"), _base_model_name(record.get("model_name", "?")))
        meta[key] = {
            "n_params": record.get("n_params"),
            "elapsed_s": round(record.get("elapsed_s", 0), 2),
            "epochs": record.get("n_epochs", 0),
        }
    return meta


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


def to_leaderboard_rows(records: list[dict]) -> list[dict]:
    """
    Publication leaderboard: prefer latest multi-seed mean ± bootstrap CI.
    Falls back to latest single-seed holdout when no summary exists.
    """
    rows: list[dict] = []
    covered: set[tuple[str, str]] = set()
    meta = _holdout_meta_by_model(records)

    summaries = latest_multi_seed_summaries(records)
    exp_with_summary = set(summaries.keys())

    for exp_id, summary in summaries.items():
        if not is_publication_experiment(exp_id):
            continue
        for model, stats in summary.items():
            key = (exp_id, model)
            covered.add(key)
            aux = meta.get(key, {})
            rows.append(
                {
                    "exp_id": exp_id,
                    "model": model,
                    "accuracy": round(stats["mean"] * 100, 1),
                    "ci_low": round(stats["ci_low"] * 100, 1),
                    "ci_high": round(stats["ci_high"] * 100, 1),
                    "std_pct": round(stats["std"] * 100, 1),
                    "n_seeds": stats.get("n_seeds", 0),
                    "loss": None,
                    "eval_set": "holdout_test",
                    "n_params": aux.get("n_params"),
                    "elapsed_s": aux.get("elapsed_s", 0),
                    "epochs": aux.get("epochs", 0),
                    "started_at": "",
                    "history": [],
                    "source": "multi_seed_summary",
                }
            )

    for record in latest_holdout_records(records):
        exp_id = record.get("exp_id", "?")
        if not is_publication_experiment(exp_id):
            continue
        if exp_id in exp_with_summary:
            continue
        key = (exp_id, _base_model_name(record.get("model_name", "?")))
        if key in covered:
            continue
        row = to_benchmark_rows([record])[0]
        row["model"] = base_model_label(row["model"])
        row["source"] = "single_seed"
        rows.append(row)

    return rows


def load_applicability_gates(records: list[dict]) -> list[dict]:
    """Return latest technique applicability record per (exp_id, technique)."""
    latest: dict[tuple[str, str], dict] = {}
    for r in records:
        if r.get("record_type") != "applicability_gate":
            continue
        key = (r.get("exp_id", "?"), r.get("technique", "?"))
        prev = latest.get(key)
        if prev is None or (r.get("started_at") or "") >= (prev.get("started_at") or ""):
            latest[key] = r

    gates = []
    for r in latest.values():
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
    """Parameter-matched table from latest holdout records only."""
    return build_param_match_table(latest_holdout_records(records))


def best_row(rows: list[dict]) -> dict | None:
    scored = [r for r in rows if r["accuracy"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: r["accuracy"])
