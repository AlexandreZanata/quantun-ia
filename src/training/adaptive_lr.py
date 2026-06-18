"""Gradient-variance-aware adaptive learning rate for QNN training."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.training.checkpoints import save_best_checkpoint
from src.training.device import resolve_device
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters, evaluate


@dataclass
class AdaptiveLRConfig:
    """Scale Adam LR from per-step gradient variance vs a target (exp_006 calibrated)."""

    base_lr: float = 0.02
    var_target: float = 0.015
    min_scale: float = 0.25
    max_scale: float = 4.0
    warmup_epochs: int = 3
    adapt_every: int = 1


def step_gradient_variance(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Gradient variance on a single forward-backward step (diagnostic from exp_006)."""
    model.zero_grad()
    pred = model(X).reshape(-1)
    target = y.reshape(-1)
    loss = criterion(pred, target)
    loss.backward()

    grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grad_parts:
        return 0.0
    grad_flat = torch.cat(grad_parts)
    if grad_flat.numel() <= 1:
        return 0.0
    return float(grad_flat.var().item())


def compute_lr_scale(grad_var: float, config: AdaptiveLRConfig) -> float:
    """Map gradient variance to LR multiplier (higher LR when variance is vanishing)."""
    if grad_var <= 0.0 or grad_var < 1e-12:
        return config.max_scale
    ratio = config.var_target / grad_var
    scale = ratio**0.5
    return float(max(config.min_scale, min(config.max_scale, scale)))


def train_model_adaptive(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    exp_id: str,
    model_name: str,
    epochs: int = 50,
    config: AdaptiveLRConfig | None = None,
    log_every: int = 10,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = True,
    device: str | None = None,
) -> ExperimentLogger:
    """Train with Adam; rescale LR from gradient variance after warmup."""
    cfg = config or AdaptiveLRConfig()
    if seed is not None:
        set_global_seed(seed)

    dev = resolve_device(device, model=model)
    model = model.to(dev)
    X = X.to(dev)
    y = y.to(dev)
    if X_test is not None:
        X_test = X_test.to(dev)
    if y_test is not None:
        y_test = y_test.to(dev)

    init_correlation_id()
    current_lr = cfg.base_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    criterion = nn.BCELoss()
    log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
    params: dict = {
        "epochs": epochs,
        "lr": cfg.base_lr,
        "adaptive_lr": True,
        "var_target": cfg.var_target,
    }
    if seed is not None:
        params["seed"] = seed
    if profile is not None:
        params["profile"] = profile
    params["device"] = str(dev)
    log._tracker.log_params(params)

    best_holdout: float | None = None
    t0 = time.time()
    for epoch in range(epochs):
        model.training = True
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()

        grad_var = 0.0
        if epoch >= cfg.warmup_epochs and epoch % cfg.adapt_every == 0:
            grad_parts = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
            if grad_parts:
                grad_flat = torch.cat(grad_parts)
                if grad_flat.numel() > 1:
                    grad_var = float(grad_flat.var().item())
            scale = compute_lr_scale(grad_var, cfg)
            current_lr = cfg.base_lr * scale
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        optimizer.step()

        with torch.no_grad():
            acc = ((pred > 0.5) == y.bool()).float().mean().item()
            epoch_metrics: dict = {
                "loss": loss.item(),
                "accuracy": acc,
                "learning_rate": current_lr,
                "grad_variance": grad_var,
            }
            if X_test is not None and y_test is not None and (
                epoch % log_every == 0 or epoch == epochs - 1
            ):
                holdout = evaluate(model, X_test, y_test)
                epoch_metrics["holdout_accuracy"] = holdout["accuracy"]
                epoch_metrics["holdout_loss"] = holdout["loss"]
                if save_checkpoints:
                    best_holdout, ckpt_path = save_best_checkpoint(
                        model,
                        exp_id,
                        model_name,
                        seed,
                        holdout["accuracy"],
                        best_metric=best_holdout,
                        higher_is_better=True,
                        config={
                            "epochs": epochs,
                            "lr": cfg.base_lr,
                            "adaptive_lr": True,
                            "var_target": cfg.var_target,
                            "seed": seed,
                            "profile": profile,
                        },
                        metadata={"holdout_accuracy": holdout["accuracy"], "epoch": epoch},
                    )
                    if ckpt_path is not None:
                        log_event(
                            "info",
                            "checkpoint saved",
                            exp_id=exp_id,
                            model_name=model_name,
                            seed=seed,
                            path=str(ckpt_path),
                            holdout_accuracy=holdout["accuracy"],
                        )
            log.log(epoch, **epoch_metrics)

        if epoch % log_every == 0:
            log_event(
                "info",
                "adaptive training progress",
                exp_id=exp_id,
                model_name=model_name,
                seed=seed,
                epoch=epoch,
                learning_rate=round(current_lr, 6),
                grad_variance=round(grad_var, 6),
                loss=round(loss.item(), 4),
                accuracy=round(acc, 3),
            )

    elapsed = time.time() - t0
    n_params = count_parameters(model)
    finish_extra: dict = {"adaptive_lr": True}
    if X_test is not None and y_test is not None:
        holdout = evaluate(model, X_test, y_test)
        finish_extra.update(
            {
                "test_accuracy": holdout["accuracy"],
                "test_loss": holdout["loss"],
                "eval_set": "holdout_test",
            }
        )
        log_event(
            "info",
            "holdout eval",
            exp_id=exp_id,
            model_name=model_name,
            seed=seed,
            test_accuracy=holdout["accuracy"],
            test_loss=holdout["loss"],
            eval_set="holdout_test",
            adaptive_lr=True,
        )
    log.finish(elapsed, n_params=n_params, **finish_extra)
    return log
