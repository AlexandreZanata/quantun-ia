"""Curriculum learning — order examples by difficulty."""

import numpy as np
import torch

from src.training.trainer import evaluate, train_model


def sort_by_difficulty(X, y, method="margin"):
    """
    method='margin': distance to class centroid (easy-first)
    method='random': random baseline
    """
    if method == "random":
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)

    ease_scores = np.array([
        np.linalg.norm(X[i] - (c0 if y[i] == 0 else c1)) * -1
        for i in range(len(X))
    ])
    idx = np.argsort(ease_scores)
    return X[idx], y[idx]


def curriculum_batches(X, y, n_stages: int = 4):
    """Split the dataset into stages of increasing difficulty."""
    X_sorted, y_sorted = sort_by_difficulty(X, y, method="margin")
    stage_size = max(len(X) // n_stages, 1)
    batches = []
    for stage in range(1, n_stages + 1):
        end = stage * stage_size if stage < n_stages else len(X)
        batches.append((X_sorted[:end], y_sorted[:end]))
    return batches


def train_curriculum_batched(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    exp_id: str,
    model_name: str,
    n_stages: int = 4,
    epochs_per_stage: int = 12,
    lr: float = 0.01,
) -> dict:
    """Train in stages (easy → hard batches) and evaluate on held-out test each stage."""
    X_t = torch.tensor(X_test)
    y_t = torch.tensor(y_test)
    stage_metrics = []

    for stage_idx, (X_stage, y_stage) in enumerate(curriculum_batches(X_train, y_train, n_stages)):
        train_model(
            model,
            torch.tensor(X_stage),
            torch.tensor(y_stage),
            exp_id,
            f"{model_name}_stage{stage_idx}",
            epochs=epochs_per_stage,
            lr=lr,
        )
        metrics = evaluate(model, X_t, y_t)
        stage_metrics.append({"stage": stage_idx, "n_samples": len(X_stage), **metrics})

    final = stage_metrics[-1] if stage_metrics else evaluate(model, X_t, y_t)
    return {"stage_metrics": stage_metrics, "test_accuracy": final["accuracy"], "test_loss": final["loss"]}
