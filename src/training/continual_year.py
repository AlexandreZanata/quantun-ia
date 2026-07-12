"""Year-by-year continual fine-tune helpers (Phase D / exp_098)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.data.continual_crop_year import rows_for_year
from src.training.batched_trainer import evaluate_with_auc, train_model_batched


def mean_backward_auc(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    evaluated_years: list[int],
) -> float:
    """Mean ROC-AUC over prior train years (forgetting probe)."""
    if not evaluated_years:
        return float("nan")
    scores: list[float] = []
    for year in evaluated_years:
        x_y, y_y = rows_for_year(x, y, years, year)
        if len(np.unique(y_y)) < 2:
            continue
        x_t = torch.tensor(x_y, dtype=torch.float32)
        y_t = torch.tensor(y_y, dtype=torch.float32)
        auc = float(evaluate_with_auc(model, x_t, y_t)["roc_auc"])
        if np.isfinite(auc):
            scores.append(auc)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def train_joint(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    profile: str,
) -> float:
    train_model_batched(
        model,
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        exp_id,
        model_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        X_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32),
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    return float(
        evaluate_with_auc(
            model,
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )["roc_auc"]
    )


def train_continual_by_year(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    years_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_years: tuple[int, ...],
    *,
    exp_id: str,
    model_name: str,
    epochs_per_year: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    profile: str,
) -> tuple[float, float]:
    """Sequential fine-tune; returns (val_auc, mean_backward_auc)."""
    seen: list[int] = []
    for year in train_years:
        x_y, y_y = rows_for_year(x_train, y_train, years_train, year)
        train_model_batched(
            model,
            torch.tensor(x_y, dtype=torch.float32),
            torch.tensor(y_y, dtype=torch.float32),
            exp_id,
            f"{model_name}_year_{year}",
            epochs=epochs_per_year,
            lr=lr,
            batch_size=min(batch_size, max(len(y_y), 1)),
            weight_decay=weight_decay,
            X_val=torch.tensor(x_val, dtype=torch.float32),
            y_val=torch.tensor(y_val, dtype=torch.float32),
            seed=seed,
            profile=profile,
            save_checkpoints=False,
        )
        seen.append(int(year))

    val_auc = float(
        evaluate_with_auc(
            model,
            torch.tensor(x_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )["roc_auc"]
    )
    # Probe forgetting on all but the last year.
    prior = seen[:-1] if len(seen) > 1 else seen
    backward = mean_backward_auc(model, x_train, y_train, years_train, prior)
    return val_auc, backward


def sklearn_auc(model, x: np.ndarray, y: np.ndarray) -> float:
    proba = model.predict_proba(x)[:, 1]
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, proba))
