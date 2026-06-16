"""Universal training loop for classical and quantum models."""

import time

import torch
import torch.nn as nn

from src.training.metrics import ExperimentLogger
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
) -> ExperimentLogger:
    init_correlation_id()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    log = ExperimentLogger(exp_id, model_name)

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
            log.log(epoch, loss=loss.item(), accuracy=acc)

        if epoch % log_every == 0:
            log_event(
                "info",
                "training progress",
                exp_id=exp_id,
                model_name=model_name,
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
            test_accuracy=holdout["accuracy"],
            test_loss=holdout["loss"],
            eval_set="holdout_test",
        )
    log.finish(elapsed, **finish_extra)
    log_event(
        "info",
        "training complete",
        exp_id=exp_id,
        model_name=model_name,
        elapsed_s=round(elapsed, 1),
        n_params=n_params,
        test_accuracy=finish_extra.get("test_accuracy"),
    )
    return log


def predict(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
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
) -> float:
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
