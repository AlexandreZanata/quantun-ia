"""Parameter-shift vs autograd ablation for deep re-upload QNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.device import resolve_device
from src.training.reproducibility import set_global_seed
from src.training.trainer import evaluate, train_model

AUTOGRAD_METHOD = "backprop"
PARAM_SHIFT_METHOD = "parameter-shift"


@dataclass(frozen=True)
class ParamShiftAblationResult:
    n_seeds: int
    mean_autograd_acc: float
    mean_param_shift_acc: float
    mean_holdout_pp: float
    autograd_grad_var: float
    param_shift_grad_var: float
    variance_ratio: float
    max_holdout_pp: float
    min_variance_ratio: float
    elapsed_s: float


def gate_passed(result: ParamShiftAblationResult) -> bool:
    return (
        result.mean_holdout_pp <= result.max_holdout_pp
        and result.variance_ratio >= result.min_variance_ratio
    )


def _gradient_variance_one_step(
    model: QuantumNetReupload,
    *,
    input_dim: int,
    batch_size: int,
) -> float:
    x_dummy = torch.randn(batch_size, input_dim)
    y_dummy = torch.randint(0, 2, (batch_size,)).float()
    model.training = True
    model.zero_grad()
    pred = model(x_dummy).reshape(-1)
    loss = nn.functional.binary_cross_entropy(pred, y_dummy.reshape(-1))
    loss.backward()
    grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        return 0.0
    grad_flat = torch.cat(grad_parts)
    if grad_flat.numel() <= 1:
        return 0.0
    return float(grad_flat.var().item())


def measure_reupload_gradient_variance(
    *,
    n_qubits: int,
    n_layers: int,
    input_dim: int,
    diff_method: str,
    n_samples: int = 10,
    seed: int | None = None,
) -> float:
    """Mean gradient variance across random initializations for one diff method."""
    batch_size = 1 if diff_method == PARAM_SHIFT_METHOD else min(10, max(input_dim, 2))
    sample_vars: list[float] = []
    for sample_idx in range(n_samples):
        if seed is not None:
            set_global_seed(seed + sample_idx)
        model = QuantumNetReupload(
            n_qubits=n_qubits,
            n_layers=n_layers,
            input_dim=input_dim,
            diff_method=diff_method,
        )
        sample_vars.append(
            _gradient_variance_one_step(model, input_dim=input_dim, batch_size=batch_size)
        )
    if not sample_vars:
        return 0.0
    return float(sum(sample_vars) / len(sample_vars))


def _train_single_sample(
    model: QuantumNetReupload,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    seed: int | None,
) -> float:
    if seed is not None:
        set_global_seed(seed)
    dev = resolve_device(None, model=model)
    model = model.to(dev)
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_holdout = x_holdout.to(dev)
    y_holdout = y_holdout.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    n = x_train.shape[0]

    for _epoch in range(epochs):
        perm = torch.randperm(n, device=dev)
        for idx in perm:
            xi = x_train[idx : idx + 1]
            yi = y_train[idx : idx + 1]
            optimizer.zero_grad()
            pred = model(xi).reshape(-1)
            yi = yi.reshape(-1)
            loss = criterion(pred, yi)
            loss.backward()
            optimizer.step()

    return float(evaluate(model, x_holdout, y_holdout)["accuracy"])


def train_reupload_grad_method(
    model: QuantumNetReupload,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    epochs: int = 20,
    lr: float = 0.02,
    seed: int | None = None,
    profile: str | None = None,
) -> float:
    """Train re-upload QNN with its configured diff method; return holdout accuracy."""
    method = model.diff_method or AUTOGRAD_METHOD
    if method == PARAM_SHIFT_METHOD:
        return _train_single_sample(
            model,
            x_train,
            y_train,
            x_holdout,
            y_holdout,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )

    train_model(
        model,
        x_train,
        y_train,
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
    return float(evaluate(model, x_holdout, y_holdout)["accuracy"])
