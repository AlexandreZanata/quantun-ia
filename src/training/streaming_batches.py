"""Chronological streaming batches for online / commit-time training."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.training.batched_trainer import train_model_batched
from src.training.trainer import predict


def chronological_batches(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_batches: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split an already time-ordered train set into contiguous batches."""
    if n_batches < 1:
        msg = "n_batches must be >= 1"
        raise ValueError(msg)
    if len(y) == 0:
        msg = "empty train set"
        raise ValueError(msg)
    n = len(y)
    size = max(n // n_batches, 1)
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_batches):
        start = i * size
        end = n if i == n_batches - 1 else min((i + 1) * size, n)
        if start >= n:
            break
        batches.append((x[start:end], y[start:end]))
    return batches


def train_streaming_batches(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    n_batches: int,
    epochs_per_batch: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    profile: str,
) -> list[float]:
    """Fine-tune chronologically; return per-batch val scores via caller metrics."""
    chunks = chronological_batches(x_train, y_train, n_batches=n_batches)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    for i, (x_b, y_b) in enumerate(chunks):
        train_model_batched(
            model,
            torch.tensor(x_b, dtype=torch.float32),
            torch.tensor(y_b, dtype=torch.float32),
            exp_id,
            f"{model_name}_batch_{i}",
            epochs=epochs_per_batch,
            lr=lr,
            batch_size=min(int(batch_size), max(len(y_b), 1)),
            weight_decay=weight_decay,
            X_val=x_val_t,
            y_val=y_val_t,
            seed=seed,
            profile=profile,
            save_checkpoints=False,
        )
    return [float(len(y_b)) for _, y_b in chunks]


def predict_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        probs = predict(model, torch.tensor(x, dtype=torch.float32)).detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float64)
