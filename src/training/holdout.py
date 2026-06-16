"""Holdout training and multi-seed evaluation helpers."""

from __future__ import annotations

import json
from datetime import datetime

import torch

from src.training import metrics as metrics_module
from src.training.statistics import paired_comparison, seed_summary
from src.training.structured_log import log_event
from src.training.trainer import train_model


def train_with_holdout(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    exp_id: str,
    model_name: str,
    epochs: int = 50,
    lr: float = 0.01,
) -> dict:
    """Train on train split; log and return metrics on held-out test split."""
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    train_model(
        model,
        X_train_t,
        y_train_t,
        exp_id,
        model_name,
        epochs=epochs,
        lr=lr,
        X_test=X_test_t,
        y_test=y_test_t,
    )
    return model.evaluate(X_test_t, y_test_t)


def _write_summary_record(record: dict) -> None:
    log_path = metrics_module.LOGS_PATH
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def summarize_multi_seed(
    exp_id: str,
    results_by_model: dict[str, list[float]],
    *,
    log_jsonl: bool = True,
) -> dict[str, dict]:
    """Log mean ± std, bootstrap CI, and optional JSONL research summary."""
    summary: dict[str, dict] = {}
    for name, accs in results_by_model.items():
        stats = seed_summary(accs)
        summary[name] = stats
        log_event(
            "info",
            "multi-seed summary",
            exp_id=exp_id,
            model=name,
            mean_holdout_acc=stats["mean"],
            std_holdout_acc=stats["std"],
            ci_low=stats["ci_low"],
            ci_high=stats["ci_high"],
            n_seeds=stats["n_seeds"],
        )

    if log_jsonl and summary:
        _write_summary_record(
            {
                "exp_id": exp_id,
                "model_name": f"{exp_id}_multi_seed_summary",
                "record_type": "multi_seed_summary",
                "started_at": datetime.now().isoformat(),
                "summary": summary,
            }
        )
    return summary


def compare_conditions(
    exp_id: str,
    condition_a: list[float],
    condition_b: list[float],
    label_a: str,
    label_b: str,
    *,
    log_jsonl: bool = True,
) -> dict:
    """Paired statistical comparison between two seed-aligned conditions."""
    comparison = paired_comparison(condition_a, condition_b)
    comparison.update({"label_a": label_a, "label_b": label_b})
    log_event(
        "info",
        "paired comparison",
        exp_id=exp_id,
        label_a=label_a,
        label_b=label_b,
        mean_diff=comparison["mean_diff"],
        p_value=comparison["p_value"],
        significant=comparison["significant"],
    )
    if log_jsonl:
        _write_summary_record(
            {
                "exp_id": exp_id,
                "model_name": f"{exp_id}_paired_{label_a}_vs_{label_b}",
                "record_type": "paired_comparison",
                "started_at": datetime.now().isoformat(),
                "comparison": comparison,
            }
        )
    return comparison
