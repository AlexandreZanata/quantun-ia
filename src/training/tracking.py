"""MLflow experiment tracking wrapper (optional, dual-write with JSONL)."""

from __future__ import annotations

import os
from typing import Any


def mlflow_enabled() -> bool:
    return os.environ.get("MLFLOW_DISABLE", "0") != "1"


class RunTracker:
    """Thin MLflow wrapper; no-ops when disabled or MLflow is unavailable."""

    def __init__(
        self,
        exp_id: str,
        model_name: str,
        *,
        seed: int | None = None,
        profile: str | None = None,
    ):
        self.exp_id = exp_id
        self.model_name = model_name
        self._active = False

        if not mlflow_enabled():
            return

        try:
            import mlflow

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)

            mlflow.set_experiment(exp_id)
            mlflow.start_run(run_name=model_name)
            tags: dict[str, str] = {"exp_id": exp_id, "model_name": model_name}
            if seed is not None:
                tags["seed"] = str(seed)
            if profile:
                tags["profile"] = profile
            mlflow.set_tags(tags)
            self._active = True
        except Exception:
            self._active = False

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            import mlflow

            safe = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()}
            mlflow.log_params(safe)
        except Exception:
            pass

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._active:
            return
        try:
            import mlflow

            safe = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            if safe:
                mlflow.log_metrics(safe, step=step)
        except Exception:
            pass

    def log_artifact(self, path: str) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.log_artifact(path)
        except Exception:
            pass

    def end(self) -> None:
        if not self._active:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass
        self._active = False
