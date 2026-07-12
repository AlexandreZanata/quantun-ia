"""Sklearn estimator wrapper implementing the TrainableMixin contract."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import ClassifierMixin

from src.training.base_model import TrainableMixin
from src.training.checkpoints import checkpoint_path
from src.training.metrics import ExperimentLogger
from src.training.reproducibility import set_global_seed
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import evaluate as torch_evaluate


def _to_numpy(X: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X)


class SklearnBinaryClassifier(TrainableMixin, nn.Module):
    """Wrap a sklearn binary classifier for holdout training and JSONL logging."""

    def __init__(self, estimator: ClassifierMixin, *, model_label: str = "sklearn"):
        super().__init__()
        self.estimator = estimator
        self.model_label = model_label
        self._fitted = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError(f"{self.model_label} is not fitted")
        X_np = _to_numpy(X)
        proba = self.estimator.predict_proba(X_np)[:, 1]
        return torch.tensor(proba, dtype=torch.float32, device=X.device)

    def count_sklearn_parameters(self) -> int:
        if hasattr(self.estimator, "coef_"):
            coef = np.asarray(self.estimator.coef_)
            intercept = np.asarray(getattr(self.estimator, "intercept_", [0]))
            return int(coef.size + intercept.size)
        if hasattr(self.estimator, "get_booster"):
            try:
                booster = self.estimator.get_booster()
                return sum(int(v) for v in booster.num_features())
            except Exception:
                return 0
        return 0

    def train(
        self,
        X: torch.Tensor | bool | None = None,
        y: torch.Tensor | None = None,
        exp_id: str = "experiment",
        model_name: str = "model",
        epochs: int = 1,
        lr: float = 0.01,
        X_test: torch.Tensor | None = None,
        y_test: torch.Tensor | None = None,
        seed: int | None = None,
        profile: str | None = None,
        save_checkpoints: bool = False,
        device: str | None = None,
    ) -> ExperimentLogger | SklearnBinaryClassifier:
        if y is None and (X is None or isinstance(X, bool)):
            self.training = True if X is None else X
            return self

        if not isinstance(X, torch.Tensor) or y is None:
            raise TypeError("train(X, y) requires torch.Tensor inputs when fitting")

        if seed is not None:
            set_global_seed(seed)

        init_correlation_id()
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)

        log = ExperimentLogger(exp_id, model_name, seed=seed, profile=profile)
        t0 = time.time()
        self.estimator.fit(X_np, y_np)
        self._fitted = True

        with torch.no_grad():
            train_t = torch.tensor(X_np, dtype=torch.float32)
            train_y = torch.tensor(y_np, dtype=torch.float32)
            train_metrics = torch_evaluate(self, train_t, train_y)
            log.log(0, loss=train_metrics["loss"], accuracy=train_metrics["accuracy"])

            finish_extra: dict[str, Any] = {}
            if X_test is not None and y_test is not None:
                test_t = torch.tensor(_to_numpy(X_test), dtype=torch.float32)
                test_y = torch.tensor(_to_numpy(y_test), dtype=torch.float32)
                holdout = torch_evaluate(self, test_t, test_y)
                finish_extra = {
                    "test_accuracy": holdout["accuracy"],
                    "test_loss": holdout["loss"],
                    "eval_set": "holdout_test",
                }
                log.log(
                    0,
                    holdout_accuracy=holdout["accuracy"],
                    holdout_loss=holdout["loss"],
                )
                if save_checkpoints:
                    self._save_sklearn_checkpoint(
                        exp_id,
                        model_name,
                        seed,
                        holdout["accuracy"],
                        config={"seed": seed, "profile": profile, "model_label": self.model_label},
                        metadata={"holdout_accuracy": holdout["accuracy"]},
                    )

        elapsed = time.time() - t0
        n_params = self.count_sklearn_parameters()
        log.finish(elapsed, n_params=n_params, **finish_extra)
        log_event(
            "info",
            "sklearn training complete",
            exp_id=exp_id,
            model_name=model_name,
            seed=seed,
            n_params=n_params,
            test_accuracy=finish_extra.get("test_accuracy"),
        )
        return log

    def _save_sklearn_checkpoint(
        self,
        exp_id: str,
        model_name: str,
        seed: int | None,
        holdout_accuracy: float,
        *,
        config: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        try:
            import joblib
        except ImportError:
            log_event(
                "warning",
                "joblib unavailable — sklearn checkpoint skipped",
                exp_id=exp_id,
                model_name=model_name,
            )
            return

        directory = checkpoint_path(exp_id, model_name, seed)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "best.joblib"
        joblib.dump(self.estimator, path)
        meta_path = directory / "config.json"
        meta_path.write_text(
            __import__("json").dumps(
                {
                    "config": config,
                    "metadata": {**metadata, "holdout_accuracy": holdout_accuracy, "format": "joblib"},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        log_event(
            "info",
            "sklearn checkpoint saved",
            exp_id=exp_id,
            model_name=model_name,
            seed=seed,
            path=str(path),
            holdout_accuracy=holdout_accuracy,
        )
