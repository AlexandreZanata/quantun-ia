"""Holdout training and multi-seed evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.training.structured_log import log_event


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

    model.train(X_train_t, y_train_t, exp_id=exp_id, model_name=model_name, epochs=epochs, lr=lr)
    metrics = model.evaluate(X_test_t, y_test_t)
    log_event(
        "info",
        "holdout eval",
        exp_id=exp_id,
        model_name=model_name,
        test_accuracy=metrics["accuracy"],
        test_loss=metrics["loss"],
        eval_set="holdout_test",
    )
    return metrics


def summarize_multi_seed(exp_id: str, results_by_model: dict[str, list[float]]) -> dict[str, dict]:
    """Log mean ± std holdout accuracy across seeds."""
    summary = {}
    for name, accs in results_by_model.items():
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[name] = {"mean": mean_acc, "std": std_acc, "runs": len(accs)}
        log_event(
            "info",
            "multi-seed summary",
            exp_id=exp_id,
            model=name,
            mean_holdout_acc=mean_acc,
            std_holdout_acc=std_acc,
            n_seeds=len(accs),
        )
    return summary
