"""Re-upload depth curriculum — grow QNN layers synchronized with margin batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.curriculum import curriculum_batches
from src.training.trainer import evaluate, train_model


def _holdout_score(
    model: QuantumNetReupload,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    *,
    metric: str,
) -> float:
    if metric == "roc_auc":
        from src.training.batched_trainer import evaluate_with_auc

        return float(evaluate_with_auc(model, x_holdout, y_holdout)["roc_auc"])
    return float(evaluate(model, x_holdout, y_holdout)["accuracy"])


def _train_stage(
    model: QuantumNetReupload,
    x_stage: np.ndarray,
    y_stage: np.ndarray,
    *,
    exp_id: str,
    model_name: str,
    epochs: int,
    lr: float,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    seed: int | None,
    profile: str | None,
    use_batched: bool,
    batch_size: int,
    weight_decay: float,
) -> None:
    x_t = torch.tensor(x_stage, dtype=torch.float32)
    y_t = torch.tensor(y_stage, dtype=torch.float32)
    if use_batched:
        from src.training.batched_trainer import train_model_batched

        train_model_batched(
            model,
            x_t,
            y_t,
            exp_id,
            model_name,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            X_val=x_holdout,
            y_val=y_holdout,
            seed=seed,
            profile=profile,
            save_checkpoints=False,
            device="cuda",
        )
    else:
        train_model(
            model,
            x_t,
            y_t,
            exp_id,
            model_name,
            epochs=epochs,
            lr=lr,
            X_test=x_holdout,
            y_test=y_holdout,
            seed=seed,
            profile=profile,
            save_checkpoints=False,
        )


def layers_for_stage(
    stage_idx: int,
    n_stages: int,
    layer_ladder: Sequence[int],
) -> int:
    """Map curriculum stage to re-upload depth (linear ladder)."""
    if n_stages <= 0:
        raise ValueError("n_stages must be positive")
    if not layer_ladder:
        raise ValueError("layer_ladder must not be empty")
    if n_stages == 1:
        return layer_ladder[-1]
    pos = stage_idx / (n_stages - 1) * (len(layer_ladder) - 1)
    idx = round(pos)
    return layer_ladder[min(max(idx, 0), len(layer_ladder) - 1)]


def _total_epochs(n_stages: int, epochs_per_stage: int) -> int:
    return n_stages * epochs_per_stage


def train_reupload_curriculum(
    model: QuantumNetReupload,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    n_stages: int = 3,
    epochs_per_stage: int = 8,
    layer_ladder: Sequence[int] = (1, 2, 3),
    lr: float = 0.02,
    seed: int | None = None,
    profile: str | None = None,
    metric: str = "accuracy",
    use_batched: bool = False,
    batch_size: int = 512,
    weight_decay: float = 1e-4,
) -> float:
    """Margin curriculum with monotonically increasing re-upload depth."""
    stages = min(n_stages, len(layer_ladder))
    for stage_idx, (x_stage, y_stage) in enumerate(curriculum_batches(x_train, y_train, stages)):
        depth = layers_for_stage(stage_idx, stages, layer_ladder)
        if depth > model.n_layers:
            model.set_n_layers(depth)
        _train_stage(
            model,
            x_stage,
            y_stage,
            exp_id=exp_id,
            model_name=f"{model_name}_stage{stage_idx}_L{depth}",
            epochs=epochs_per_stage,
            lr=lr,
            x_holdout=x_holdout,
            y_holdout=y_holdout,
            seed=seed,
            profile=profile,
            use_batched=use_batched,
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
    return _holdout_score(model, x_holdout, y_holdout, metric=metric)


def train_fixed_reupload(
    build_model: Callable[[int], QuantumNetReupload],
    max_layers: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    n_stages: int = 3,
    epochs_per_stage: int = 8,
    lr: float = 0.02,
    seed: int | None = None,
    profile: str | None = None,
    metric: str = "accuracy",
    use_batched: bool = False,
    batch_size: int = 512,
    weight_decay: float = 1e-4,
) -> float:
    """Epoch- and curriculum-matched fixed-depth baseline."""
    model = build_model(max_layers)
    for stage_idx, (x_stage, y_stage) in enumerate(curriculum_batches(x_train, y_train, n_stages)):
        _train_stage(
            model,
            x_stage,
            y_stage,
            exp_id=exp_id,
            model_name=f"{model_name}_stage{stage_idx}_L{max_layers}",
            epochs=epochs_per_stage,
            lr=lr,
            x_holdout=x_holdout,
            y_holdout=y_holdout,
            seed=seed,
            profile=profile,
            use_batched=use_batched,
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
    return _holdout_score(model, x_holdout, y_holdout, metric=metric)
