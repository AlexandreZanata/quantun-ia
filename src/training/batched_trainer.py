"""Mini-batch training loop for large tabular models on RTX 4060."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.training.checkpoints import save_best_checkpoint
from src.training.device import resolve_device
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters, evaluate, predict


def evaluate_with_auc(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Return accuracy, loss, and ROC-AUC for binary predictions."""
    metrics = evaluate(model, X, y)
    with torch.no_grad():
        probs = predict(model, X).detach().cpu().numpy()
    labels = y.detach().cpu().numpy()
    if len(np.unique(labels)) < 2:
        metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics


def evaluate_with_auc_batched(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 8192,
) -> dict:
    """ROC-AUC on large validation tensors without full-batch GPU OOM."""
    if len(X) <= batch_size:
        return evaluate_with_auc(model, X, y)

    model.eval()
    dev = next(model.parameters()).device
    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = X[start : start + batch_size].to(dev)
            y_batch = y[start : start + batch_size].to(dev)
            prob_chunks.append(predict(model, x_batch).detach().cpu().numpy())
            label_chunks.append(y_batch.detach().cpu().numpy())

    probs = np.concatenate(prob_chunks)
    labels = np.concatenate(label_chunks)
    preds = probs > 0.5
    accuracy = float(np.mean(preds == labels.astype(bool)))
    loss = float(np.mean((probs - labels) ** 2))
    if len(np.unique(labels)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(labels, probs))
    return {"accuracy": accuracy, "loss": loss, "roc_auc": auc}


def train_model_batched(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    exp_id: str,
    model_name: str,
    *,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 2048,
    weight_decay: float = 1e-4,
    log_every: int = 1,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = False,
    device: str | None = None,
) -> ExperimentLogger:
    """Train with mini-batches to avoid OOM on million-row tabular sets."""
    if seed is not None:
        set_global_seed(seed)

    dev = resolve_device(device, model=model)
    model = model.to(dev)
    if X_val is not None:
        X_val = X_val.to(dev)
    if y_val is not None:
        y_val = y_val.to(dev)

    init_correlation_id()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
    log._tracker.log_params(
        {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
            "profile": profile,
            "device": str(dev),
        }
    )

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    eval_batch_size = 8192

    def _val_metrics() -> dict:
        if X_val is None or y_val is None:
            return {}
        if len(y_val) > eval_batch_size:
            return evaluate_with_auc_batched(model, X_val, y_val, batch_size=eval_batch_size)
        return evaluate_with_auc(model, X_val, y_val)

    best_val_auc: float | None = None
    t0 = time.time()
    for epoch in range(epochs):
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

        epoch_metrics = {
            "loss": epoch_loss / max(n_batches, 1),
            "accuracy": epoch_acc / max(n_batches, 1),
        }
        if X_val is not None and y_val is not None and (epoch % log_every == 0 or epoch == epochs - 1):
            val_metrics = _val_metrics()
            epoch_metrics["val_accuracy"] = val_metrics["accuracy"]
            epoch_metrics["val_loss"] = val_metrics["loss"]
            epoch_metrics["val_roc_auc"] = val_metrics["roc_auc"]
            if save_checkpoints:
                best_val_auc, ckpt_path = save_best_checkpoint(
                    model,
                    exp_id,
                    model_name,
                    seed,
                    val_metrics["roc_auc"],
                    best_metric=best_val_auc,
                    higher_is_better=True,
                    config={
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                        "profile": profile,
                    },
                    metadata={"val_roc_auc": val_metrics["roc_auc"], "epoch": epoch},
                )
                if ckpt_path is not None:
                    log_event(
                        "info",
                        "checkpoint saved",
                        exp_id=exp_id,
                        model_name=model_name,
                        seed=seed,
                        path=str(ckpt_path),
                        val_roc_auc=val_metrics["roc_auc"],
                    )
        log.log(epoch, **epoch_metrics)

    elapsed = time.time() - t0
    n_params = count_parameters(model)
    finish_extra: dict = {"n_params": n_params}
    if X_val is not None and y_val is not None:
        val_metrics = _val_metrics()
        finish_extra.update(
            {
                "test_accuracy": val_metrics["accuracy"],
                "test_loss": val_metrics["loss"],
                "val_roc_auc": val_metrics["roc_auc"],
                "eval_set": "validation",
            }
        )
    log.finish(elapsed, **finish_extra)
    log_event(
        "info",
        "batched training complete",
        exp_id=exp_id,
        model_name=model_name,
        seed=seed,
        elapsed_s=round(elapsed, 1),
        n_params=n_params,
        val_roc_auc=finish_extra.get("val_roc_auc"),
    )
    return log
