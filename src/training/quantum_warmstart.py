"""Classical-first warm-start schedule for hybrid sandwich models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.quantum.hybrid_model import HybridSandwich
from src.training.batched_trainer import train_model_batched


@dataclass(frozen=True)
class WarmStartConfig:
    """Split total epochs into classical-only and quantum-enabled phases."""

    classical_fraction: float = 0.7
    total_epochs: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.classical_fraction < 1.0:
            raise ValueError("classical_fraction must be in (0, 1)")
        if self.total_epochs < 2:
            raise ValueError("total_epochs must be >= 2 for warm-start")


def split_warmstart_epochs(total_epochs: int, classical_fraction: float) -> tuple[int, int]:
    """Return (classical_epochs, quantum_epochs) that sum to total_epochs."""
    WarmStartConfig(classical_fraction=classical_fraction, total_epochs=total_epochs)
    classical_epochs = max(1, int(total_epochs * classical_fraction))
    quantum_epochs = max(1, total_epochs - classical_epochs)
    if classical_epochs + quantum_epochs > total_epochs:
        classical_epochs = total_epochs - quantum_epochs
    return classical_epochs, quantum_epochs


def _set_qlayer_trainable(model: HybridSandwich, *, enabled: bool) -> None:
    for param in model.qlayer.parameters():
        param.requires_grad = enabled


def train_hybrid_warmstart(
    model: HybridSandwich,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    config: WarmStartConfig,
    lr: float = 0.01,
    batch_size: int = 512,
    weight_decay: float = 1e-4,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = False,
    device: str | None = None,
) -> tuple[int, int]:
    """
    Train hybrid sandwich classical-first, then enable the QNN block.

    Phase 1 keeps the QNN in the forward pass (frozen) so the post head sees
    the same feature geometry as phase 2. Phase 2 unfreezes QNN weights.

    Returns (classical_epochs, quantum_epochs) executed.
    """
    classical_epochs, quantum_epochs = split_warmstart_epochs(
        config.total_epochs,
        config.classical_fraction,
    )
    train_kwargs = dict(
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=save_checkpoints,
        device=device,
    )

    model.set_quantum_enabled(True)
    _set_qlayer_trainable(model, enabled=False)
    train_model_batched(
        model,
        x_train,
        y_train,
        exp_id,
        f"{model_name}_classical",
        epochs=classical_epochs,
        **train_kwargs,
    )

    _set_qlayer_trainable(model, enabled=True)
    train_model_batched(
        model,
        x_train,
        y_train,
        exp_id,
        f"{model_name}_quantum",
        epochs=quantum_epochs,
        **train_kwargs,
    )

    return classical_epochs, quantum_epochs
