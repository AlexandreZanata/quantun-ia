"""Curriculum learning — order examples by difficulty."""

import time

import numpy as np
import torch
import torch.nn as nn

from src.training.metrics import ExperimentLogger
from src.training.trainer import count_parameters, evaluate


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

    # Smaller distance to own centroid = easier example; sort ascending = easy-first.
    ease_scores = np.array([
        np.linalg.norm(X[i] - (c0 if y[i] == 0 else c1))
        for i in range(len(X))
    ])
    idx = np.argsort(ease_scores)
    return X[idx], y[idx]


def curriculum_batches(X, y, n_stages: int = 4):
    """Split the dataset into cumulative stages of increasing difficulty."""
    X_sorted, y_sorted = sort_by_difficulty(X, y, method="margin")
    stage_size = max(len(X) // n_stages, 1)
    batches = []
    for stage in range(1, n_stages + 1):
        end = stage * stage_size if stage < n_stages else len(X)
        batches.append((X_sorted[:end], y_sorted[:end]))
    return batches


def _train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    log: ExperimentLogger | None = None,
    epoch_offset: int = 0,
) -> None:
    for epoch in range(epochs):
        model.training = True
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        if log is not None:
            with torch.no_grad():
                acc = ((pred > 0.5) == y.bool()).float().mean().item()
            log.log(epoch_offset + epoch, loss=loss.item(), accuracy=acc)


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
    refine_epochs: int = 12,
) -> dict:
    """Train in stages (easy → hard batches) with a shared optimizer and final full-data refine."""
    X_t = torch.tensor(X_test)
    y_t = torch.tensor(y_test)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = ExperimentLogger(exp_id, model_name)
    stage_metrics = []
    epoch_offset = 0
    t0 = time.time()

    for stage_idx, (X_stage, y_stage) in enumerate(curriculum_batches(X_train, y_train, n_stages)):
        X_stage_t = torch.tensor(X_stage)
        y_stage_t = torch.tensor(y_stage)
        _train_epochs(
            model,
            optimizer,
            criterion,
            X_stage_t,
            y_stage_t,
            epochs_per_stage,
            log=log,
            epoch_offset=epoch_offset,
        )
        epoch_offset += epochs_per_stage
        metrics = evaluate(model, X_t, y_t)
        stage_metrics.append(
            {
                "stage": stage_idx,
                "n_samples": len(X_stage),
                "holdout_accuracy": metrics["accuracy"],
                **metrics,
            }
        )

    # Final refinement on the full training set to recover from partial-stage bias.
    _train_epochs(
        model,
        optimizer,
        criterion,
        torch.tensor(X_train),
        torch.tensor(y_train),
        refine_epochs,
        log=log,
        epoch_offset=epoch_offset,
    )

    final = evaluate(model, X_t, y_t)
    elapsed = time.time() - t0
    log.finish(
        elapsed,
        test_accuracy=final["accuracy"],
        test_loss=final["loss"],
        eval_set="holdout_test",
        n_params=count_parameters(model),
        curriculum_stages=n_stages,
        stage_holdout_metrics=stage_metrics,
    )
    return {
        "stage_metrics": stage_metrics,
        "test_accuracy": final["accuracy"],
        "test_loss": final["loss"],
    }
