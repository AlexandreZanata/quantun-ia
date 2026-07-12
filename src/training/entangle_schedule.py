"""Dynamic entanglement schedule for QuantumNetEntangled (none → chain → ring)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.trainer import evaluate, predict, train_model

_HOLDOUT_METRICS = frozenset({"accuracy", "pr_auc", "roc_auc"})


def holdout_score(
    model: QuantumNetEntangled,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    *,
    metric: str = "accuracy",
) -> float:
    """Return holdout accuracy, PR-AUC, or ROC-AUC for a trained entangled QNN."""
    if metric not in _HOLDOUT_METRICS:
        raise ValueError(f"metric must be one of {sorted(_HOLDOUT_METRICS)}")
    if metric == "pr_auc":
        from src.application.balanced_metrics import pr_auc

        device = next(model.parameters()).device
        with torch.no_grad():
            probs = predict(model, x_holdout.to(device)).detach().cpu().numpy()
        labels = y_holdout.detach().cpu().numpy()
        score = pr_auc(labels, probs)
        return float(score) if score is not None else 0.5
    if metric == "roc_auc":
        from src.training.batched_trainer import evaluate_with_auc

        return float(evaluate_with_auc(model, x_holdout, y_holdout)["roc_auc"])
    return float(evaluate(model, x_holdout, y_holdout)["accuracy"])

DEFAULT_ENTANGLEMENT_LADDER: tuple[str, ...] = ("none", "chain", "ring")


def entanglement_for_stage(
    stage_idx: int,
    n_stages: int,
    ladder: Sequence[str] = DEFAULT_ENTANGLEMENT_LADDER,
) -> str:
    """Map curriculum stage index to entanglement topology (linear ladder)."""
    if n_stages <= 0:
        raise ValueError("n_stages must be positive")
    if not ladder:
        raise ValueError("ladder must not be empty")
    if n_stages == 1:
        return ladder[-1]
    pos = stage_idx / (n_stages - 1) * (len(ladder) - 1)
    idx = round(pos)
    return ladder[min(max(idx, 0), len(ladder) - 1)]


def _total_schedule_epochs(n_stages: int, epochs_per_stage: int) -> int:
    return n_stages * epochs_per_stage


def train_entangled_schedule(
    build_model: Callable[[str], QuantumNetEntangled],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    n_stages: int = 5,
    epochs_per_stage: int = 10,
    ladder: Sequence[str] = DEFAULT_ENTANGLEMENT_LADDER,
    lr: float = 0.02,
    seed: int | None = None,
    profile: str | None = None,
    metric: str = "accuracy",
) -> float:
    """
    Train with increasing entanglement over stages; return holdout score.

    Each stage swaps the PennyLane topology and continues from prior weights.
    """
    if n_stages <= 0 or epochs_per_stage <= 0:
        raise ValueError("n_stages and epochs_per_stage must be positive")

    ent0 = entanglement_for_stage(0, n_stages, ladder)
    model = build_model(ent0)

    for stage_idx in range(n_stages):
        ent = entanglement_for_stage(stage_idx, n_stages, ladder)
        if stage_idx > 0 and ent != model.entanglement:
            model.set_entanglement(ent)

        train_model(
            model,
            x_train,
            y_train,
            exp_id,
            f"{model_name}_stage{stage_idx}_{ent}",
            epochs=epochs_per_stage,
            lr=lr,
            X_test=x_holdout,
            y_test=y_holdout,
            seed=seed,
            profile=profile,
            save_checkpoints=False,
        )

    return holdout_score(model, x_holdout, y_holdout, metric=metric)


def train_fixed_entangled(
    build_model: Callable[[str], QuantumNetEntangled],
    entanglement: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    n_stages: int,
    epochs_per_stage: int,
    lr: float = 0.02,
    seed: int | None = None,
    profile: str | None = None,
    metric: str = "accuracy",
) -> float:
    """Epoch-matched fixed-topology baseline (same total epochs as schedule)."""
    total_epochs = _total_schedule_epochs(n_stages, epochs_per_stage)
    model = build_model(entanglement)
    train_model(
        model,
        x_train,
        y_train,
        exp_id,
        model_name,
        epochs=total_epochs,
        lr=lr,
        X_test=x_holdout,
        y_test=y_holdout,
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    return holdout_score(model, x_holdout, y_holdout, metric=metric)
