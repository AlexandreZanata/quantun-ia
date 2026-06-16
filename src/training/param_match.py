"""Parameter-count utilities and parameter-matched classical baselines."""

from __future__ import annotations

import re

from src.classical.mlp import ClassicalNet
from src.training.trainer import count_parameters


def classical_n_params(hidden: int, input_dim: int = 2) -> int:
    """Trainable parameter count for ClassicalNet(input_dim, hidden)."""
    model = ClassicalNet(input_dim=input_dim, hidden=hidden)
    return count_parameters(model)


def nearest_classical_hidden(target_n_params: int, input_dim: int = 2, max_hidden: int = 256) -> int:
    """Return hidden size whose ClassicalNet param count is closest to target."""
    best_hidden = 8
    best_diff = float("inf")
    for hidden in range(1, max_hidden + 1):
        diff = abs(classical_n_params(hidden, input_dim) - target_n_params)
        if diff < best_diff:
            best_diff = diff
            best_hidden = hidden
    return best_hidden


def build_param_matched_classical(target_n_params: int, input_dim: int = 2) -> ClassicalNet:
    hidden = nearest_classical_hidden(target_n_params, input_dim)
    return ClassicalNet(input_dim=input_dim, hidden=hidden)


_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _base_model_name(model_name: str) -> str:
    return _SEED_SUFFIX.sub("", model_name)


def latest_records_by_model(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Keep the latest JSONL record per (exp_id, model_name without seed suffix)."""
    latest: dict[tuple[str, str], dict] = {}
    for record in records:
        if record.get("record_type"):
            continue
        exp_id = record.get("exp_id", "?")
        base = _base_model_name(record.get("model_name", "?"))
        latest[(exp_id, base)] = record
    return latest


def build_param_match_table(records: list[dict]) -> list[dict]:
    """
    Build a comparison table: each model with n_params, holdout accuracy,
    and the nearest parameter-matched classical baseline.
    """
    latest = latest_records_by_model(records)
    rows: list[dict] = []

    for (exp_id, model), record in sorted(latest.items()):
        n_params = record.get("n_params")
        acc = record.get("test_accuracy")
        if acc is None:
            acc = record.get("final_acc")
        if n_params is None or acc is None:
            continue

        matched_hidden = nearest_classical_hidden(int(n_params))
        matched_n = classical_n_params(matched_hidden)
        matched_key = (exp_id, f"classical_matched_h{matched_hidden}")
        matched_record = latest.get(matched_key)
        matched_acc = None
        if matched_record:
            matched_acc = matched_record.get("test_accuracy") or matched_record.get("final_acc")

        rows.append(
            {
                "exp_id": exp_id,
                "model": model,
                "n_params": int(n_params),
                "accuracy_pct": round(acc * 100, 1),
                "matched_classical_hidden": matched_hidden,
                "matched_n_params": matched_n,
                "matched_accuracy_pct": round(matched_acc * 100, 1) if matched_acc is not None else None,
                "param_delta": int(n_params) - matched_n,
            }
        )
    return rows
