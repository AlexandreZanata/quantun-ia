"""Probability calibration for human-facing risk scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CalibrationArtifact:
    method: str
    model_payload: dict[str, Any]

    def transform(self, probs: list[float] | np.ndarray) -> list[float]:
        values = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
        if self.method == "isotonic":
            x_thresholds = np.asarray(self.model_payload["x_thresholds"], dtype=float)
            y_thresholds = np.asarray(self.model_payload["y_thresholds"], dtype=float)
            calibrated = np.interp(values, x_thresholds, y_thresholds)
            return np.clip(calibrated, 0.0, 1.0).tolist()
        if self.method == "platt":
            lr = LogisticRegression()
            lr.classes_ = np.asarray(self.model_payload["classes"])
            lr.coef_ = np.asarray(self.model_payload["coef"])
            lr.intercept_ = np.asarray(self.model_payload["intercept"])
            logits = lr.decision_function(values.reshape(-1, 1))
            calibrated = 1.0 / (1.0 + np.exp(-logits))
            return calibrated.tolist()
        msg = f"unknown calibration method: {self.method}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "model_payload": self.model_payload}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationArtifact:
        return cls(method=str(payload["method"]), model_payload=dict(payload["model_payload"]))


def fit_isotonic_calibrator(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> CalibrationArtifact:
    model = IsotonicRegression(out_of_bounds="clip")
    x = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=int)
    model.fit(x, y)
    return CalibrationArtifact(
        method="isotonic",
        model_payload={
            "x_thresholds": model.X_thresholds_.tolist(),
            "y_thresholds": model.y_thresholds_.tolist(),
        },
    )


def fit_platt_calibrator(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> CalibrationArtifact:
    model = LogisticRegression(max_iter=200)
    x = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    model.fit(x, y)
    return CalibrationArtifact(
        method="platt",
        model_payload={
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "classes": model.classes_.tolist(),
        },
    )


def apply_isotonic(
    probs: list[float] | np.ndarray,
    artifact: CalibrationArtifact,
) -> list[float]:
    return artifact.transform(probs)


def spearman_rank_correlation(
    left: list[float] | np.ndarray,
    right: list[float] | np.ndarray,
) -> float:
    rho, _ = spearmanr(left, right)
    if np.isnan(rho):
        return 0.0
    return float(rho)
