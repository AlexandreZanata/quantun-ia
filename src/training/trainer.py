"""Universal training loop for classical and quantum models."""

import time

import torch
import torch.nn as nn

from src.training.checkpoints import save_best_checkpoint
from src.training.device import resolve_device
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.structured_log import init_correlation_id, log_event


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    exp_id: str,
    model_name: str,
    epochs: int = 50,
    lr: float = 0.01,
    log_every: int = 10,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    seed: int | None = None,
    profile: str | None = None,
    save_checkpoints: bool = True,
    device: str | None = None,
) -> ExperimentLogger:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
    params: dict = {"epochs": epochs, "lr": lr}
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
        optimizer.step()

        with torch.no_grad():
            acc = ((pred > 0.5) == y.bool()).float().mean().item()
            epoch_metrics: dict = {"loss": loss.item(), "accuracy": acc}
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
                        config={"epochs": epochs, "lr": lr, "seed": seed, "profile": profile},
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
                "training progress",
                exp_id=exp_id,
                model_name=model_name,
                seed=seed,
                epoch=epoch,
                loss=round(loss.item(), 4),
                accuracy=round(acc, 3),
            )

    elapsed = time.time() - t0
    n_params = count_parameters(model)
    finish_extra: dict = {}
    if X_test is not None and y_test is not None:
        holdout = evaluate(model, X_test, y_test)
        finish_extra = {
            "test_accuracy": holdout["accuracy"],
            "test_loss": holdout["loss"],
            "eval_set": "holdout_test",
        }
        log_event(
            "info",
            "holdout eval",
            exp_id=exp_id,
            model_name=model_name,
            seed=seed,
            test_accuracy=holdout["accuracy"],
            test_loss=holdout["loss"],
            eval_set="holdout_test",
        )
    log.finish(elapsed, n_params=n_params, **finish_extra)
    log_event(
        "info",
        "training complete",
        exp_id=exp_id,
        model_name=model_name,
        seed=seed,
        elapsed_s=round(elapsed, 1),
        n_params=n_params,
        test_accuracy=finish_extra.get("test_accuracy"),
    )
    return log


def _tensor_device(model: nn.Module) -> torch.device:
    from src.training.device import model_requires_cpu

    if model_requires_cpu(model):
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    X = X.to(_tensor_device(model))
    with torch.no_grad():
        return model(X)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    dev = _tensor_device(model)
    X = X.to(dev)
    y = y.to(dev)
    model.eval()
    with torch.no_grad():
        pred = model(X)
        acc = ((pred > 0.5) == y.bool()).float().mean().item()
        loss = nn.BCELoss()(pred, y).item()
    return {"accuracy": acc, "loss": loss}


def fine_tune(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 20,
    lr: float = 0.01,
    seed: int | None = None,
    device: str | None = None,
) -> float:
    if seed is not None:
        set_global_seed(seed)

    dev = resolve_device(device, model=model)
    model = model.to(dev)
    X, y = X.to(dev), y.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        model.training = True
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    return evaluate(model, X, y)["accuracy"]
