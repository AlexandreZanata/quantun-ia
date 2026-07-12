"""SPEI-proxy curriculum ordering for agro tabular training (Phase D / exp_097)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.open_acyd import SOIL_COLUMNS, WEATHER_PREFIXES
from src.training.batched_trainer import evaluate_with_auc
from src.training.device import resolve_device
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.trainer import count_parameters

# ACYD 37-d layout: lat, lon, log(area), 6 soils, then weather blocks (mean,std,min,max).
ACYD_PRECIP_MEAN_FEATURE_INDEX = 3 + len(SOIL_COLUMNS)
assert WEATHER_PREFIXES[0] == "precipitation_week_"
assert ACYD_PRECIP_MEAN_FEATURE_INDEX == 9


def spei_proxy_difficulty(x: np.ndarray, *, precip_col: int = ACYD_PRECIP_MEAN_FEATURE_INDEX) -> np.ndarray:
    """Higher score = harder (drier). Uses −precip_mean so wet seasons sort first."""
    features = np.asarray(x, dtype=np.float64)
    if features.ndim != 2:
        msg = f"expected 2-d features, got shape {features.shape}"
        raise ValueError(msg)
    if precip_col < 0 or precip_col >= features.shape[1]:
        msg = f"precip_col={precip_col} out of range for width {features.shape[1]}"
        raise ValueError(msg)
    return -features[:, precip_col]


def sort_by_spei_difficulty(
    x: np.ndarray,
    y: np.ndarray,
    *,
    precip_col: int = ACYD_PRECIP_MEAN_FEATURE_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort rows easy→hard by SPEI proxy (wet first, dry last)."""
    difficulty = spei_proxy_difficulty(x, precip_col=precip_col)
    idx = np.argsort(difficulty, kind="mergesort")
    return x[idx], y[idx]


def sort_by_random_order(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    return x[idx], y[idx]


def cumulative_curriculum_stages(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_stages: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cumulative prefixes of an already-sorted (easy→hard or random) train set."""
    if n_stages < 1:
        msg = "n_stages must be >= 1"
        raise ValueError(msg)
    stage_size = max(len(x) // n_stages, 1)
    stages: list[tuple[np.ndarray, np.ndarray]] = []
    for stage in range(1, n_stages + 1):
        end = stage * stage_size if stage < n_stages else len(x)
        stages.append((x[:end], y[:end]))
    return stages


def train_staged_curriculum_batched(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    n_stages: int = 4,
    epochs_per_stage: int = 2,
    refine_epochs: int = 4,
    lr: float = 0.001,
    batch_size: int = 2048,
    weight_decay: float = 1e-4,
    seed: int | None = None,
    profile: str | None = None,
) -> dict:
    """Mini-batch staged curriculum with shared Adam + final full-data refine."""
    if seed is not None:
        set_global_seed(seed)

    dev = resolve_device(model=model)
    model = model.to(dev)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
    log._tracker.log_params(
        {
            "n_stages": n_stages,
            "epochs_per_stage": epochs_per_stage,
            "refine_epochs": refine_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
            "profile": profile,
            "device": str(dev),
        }
    )

    stages = cumulative_curriculum_stages(x_train, y_train, n_stages=n_stages)
    epoch = 0
    t0 = time.time()
    stage_metrics: list[dict] = []

    def _run_epochs(x_np: np.ndarray, y_np: np.ndarray, n_epochs: int) -> None:
        nonlocal epoch
        loader = DataLoader(
            TensorDataset(
                torch.tensor(x_np, dtype=torch.float32),
                torch.tensor(y_np, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        for _ in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += ((pred > 0.5) == y_batch.bool()).float().mean().item()
                n_batches += 1
            val = evaluate_with_auc(model, x_val_t, y_val_t)
            log.log(
                epoch,
                loss=epoch_loss / max(n_batches, 1),
                accuracy=epoch_acc / max(n_batches, 1),
                val_accuracy=val["accuracy"],
                val_loss=val["loss"],
                val_roc_auc=val["roc_auc"],
            )
            epoch += 1

    for stage_idx, (x_stage, y_stage) in enumerate(stages):
        _run_epochs(x_stage, y_stage, epochs_per_stage)
        val = evaluate_with_auc(model, x_val_t, y_val_t)
        stage_metrics.append(
            {
                "stage": stage_idx,
                "n_samples": int(len(y_stage)),
                "val_roc_auc": float(val["roc_auc"]),
            }
        )

    _run_epochs(x_train, y_train, refine_epochs)
    final = evaluate_with_auc(model, x_val_t, y_val_t)
    elapsed = time.time() - t0
    log.finish(
        elapsed,
        test_accuracy=final["accuracy"],
        test_loss=final["loss"],
        eval_set="validation",
        n_params=count_parameters(model),
        curriculum_stages=n_stages,
        stage_holdout_metrics=stage_metrics,
        val_roc_auc=final["roc_auc"],
    )
    return {
        "stage_metrics": stage_metrics,
        "val_roc_auc": float(final["roc_auc"]),
        "val_accuracy": float(final["accuracy"]),
        "elapsed_s": float(elapsed),
        "n_params": count_parameters(model),
    }
