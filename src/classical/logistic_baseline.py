"""Logistic regression baseline for tabular classification."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.classical.sklearn_wrapper import SklearnBinaryClassifier


def LogisticBaseline(
    input_dim: int,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> SklearnBinaryClassifier:
    """Clinical-style logistic regression baseline (input_dim unused — sklearn infers features)."""
    _ = input_dim
    estimator = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    return SklearnBinaryClassifier(estimator, model_label="logistic_regression")
