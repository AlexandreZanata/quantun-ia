"""Shallow XGBoost baseline for tabular classification."""

from __future__ import annotations

from src.classical.sklearn_wrapper import SklearnBinaryClassifier


def XGBoostShallow(
    input_dim: int,
    *,
    max_depth: int = 3,
    n_estimators: int = 50,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> SklearnBinaryClassifier:
    """Shallow gradient-boosted trees — strong tabular baseline at small depth."""
    _ = input_dim
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoostShallow — pip install xgboost") from exc

    estimator = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
    )
    return SklearnBinaryClassifier(estimator, model_label="xgboost_shallow")
