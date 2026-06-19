"""Base mixin enforcing the model interface contract from .cursor/rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

    from src.training.metrics import ExperimentLogger


class TrainableMixin:
    """Mixin: train(X, y), predict(X), evaluate(X, y) -> dict.

    Also supports PyTorch mode toggling via train(False) / train(True).
    """

    def train(
        self,
        X: Union[torch.Tensor, bool, None] = None,
        y: torch.Tensor | None = None,
        exp_id: str = "experiment",
        model_name: str = "model",
        epochs: int = 50,
        lr: float = 0.01,
        X_test: torch.Tensor | None = None,
        y_test: torch.Tensor | None = None,
    ) -> ExperimentLogger | "TrainableMixin":
        if y is None and (X is None or isinstance(X, bool)):
            mode = True if X is None else X
            super().train(mode)  # type: ignore[misc]
            return self

        from src.training.trainer import train_model

        return train_model(
            self, X, y, exp_id, model_name, epochs=epochs, lr=lr, X_test=X_test, y_test=y_test
        )

    def eval(self) -> "TrainableMixin":
        super().eval()  # type: ignore[misc]
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        from src.training.trainer import predict

        return predict(self, X)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        from src.training.trainer import evaluate

        return evaluate(self, X, y)
