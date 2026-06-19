"""Evaluate published serve checkpoints on open dataset holdout splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.application.batch_predict import BatchPredictDTO, run_batch_predict
from src.application.open_serve import (
    DEFAULT_HYBRID_SERVE_EXP_ID,
    DEFAULT_HYBRID_SERVE_MODEL,
    DEFAULT_SERVE_DATASET,
    DEFAULT_SERVE_EXP_ID,
    DEFAULT_SERVE_MODEL,
    DEFAULT_SERVE_SEED,
    ensure_large_nano_hybrid_serve_artifact,
    ensure_large_nano_serve_artifact,
    open_dataset_feature_count,
)
from src.data.open_manifest import get_dataset, load_manifest
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.structured_log import init_correlation_id, log_event

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHUNK_SIZE = 2048


@dataclass(frozen=True)
class ServeModelProfile:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    label: str


SERVE_MODELS: tuple[ServeModelProfile, ...] = (
    ServeModelProfile(
        exp_id=DEFAULT_SERVE_EXP_ID,
        model_name=DEFAULT_SERVE_MODEL,
        dataset=DEFAULT_SERVE_DATASET,
        seed=DEFAULT_SERVE_SEED,
        label="LargeNanoMLP — HIGGS",
    ),
    ServeModelProfile(
        exp_id=DEFAULT_HYBRID_SERVE_EXP_ID,
        model_name=DEFAULT_HYBRID_SERVE_MODEL,
        dataset=DEFAULT_SERVE_DATASET,
        seed=DEFAULT_SERVE_SEED,
        label="LargeNano Hybrid — HIGGS",
    ),
    ServeModelProfile(
        exp_id="exp_034",
        model_name=DEFAULT_SERVE_MODEL,
        dataset="synthea_cv_risk_v1",
        seed=DEFAULT_SERVE_SEED,
        label="LargeNanoMLP — Synthea CV",
    ),
)


@dataclass(frozen=True)
class EvaluateServeModelError:
    code: str
    message: str


@dataclass(frozen=True)
class EvaluateServeModelDTO:
    exp_id: str
    model_name: str
    dataset: str
    seed: int = DEFAULT_SERVE_SEED
    split: str = "val"
    n_rows: int | None = 5000
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass(frozen=True)
class ConfusionStats:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


@dataclass(frozen=True)
class EvaluateServeModelResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    split: str
    n_rows: int
    accuracy: float
    roc_auc: float
    brier_score: float
    confusion: ConfusionStats
    checkpoint_path: str
    mean_probability: float
    positive_rate: float
    fpr: list[float]
    tpr: list[float]
    sample_rows: list[dict[str, float | int]]
    record_source: str = "evaluate_serve_model"


def list_serve_models() -> list[ServeModelProfile]:
    return list(SERVE_MODELS)


def load_open_split_labeled(
    dataset_id: str,
    root: Path,
    *,
    split: str = "val",
    n_rows: int | None = None,
    random_state: int = 42,
    min_negatives: int = 0,
    force_balanced: bool = False,
) -> tuple[list[list[float]], list[int]]:
    """Load raw (unscaled) features and integer labels from an open dataset split."""
    if split not in {"train", "val", "test"}:
        msg = f"split must be train|val|test, got {split}"
        raise ValueError(msg)

    manifest = load_manifest(root / "data" / "open" / "manifest.json")
    dataset = get_dataset(manifest, dataset_id)
    if not dataset.get("ready"):
        msg = f"{dataset_id} is not ready"
        raise ValueError(msg)

    processed = root / "data" / "open" / dataset["path"]
    feature_cols = [f"feature_{i}" for i in range(int(dataset["n_features"]))]
    frame = pd.read_parquet(processed / dataset["files"][split])
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=int)

    if n_rows is not None and n_rows < len(labels):
        indices = np.arange(len(labels))
        stratified, _ = train_test_split(
            indices,
            train_size=n_rows,
            stratify=labels,
            random_state=random_state,
        )
        stratified_neg = int(np.sum(labels[stratified] == 0))
        if force_balanced and min_negatives > 0:
            from src.application.balanced_metrics import balanced_subsample_indices

            selected = balanced_subsample_indices(
                labels,
                n_rows,
                min_negatives=min_negatives,
                random_state=random_state,
            )
        elif stratified_neg == 0 and min_negatives > 0:
            from src.application.balanced_metrics import balanced_subsample_indices

            selected = balanced_subsample_indices(
                labels,
                n_rows,
                min_negatives=min_negatives,
                random_state=random_state,
            )
        else:
            selected = stratified
        features = features[selected]
        labels = labels[selected]

    return features.tolist(), labels.tolist()


def _confusion_stats(y_true: list[int], y_pred: list[int]) -> ConfusionStats:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ConfusionStats(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def _ensure_serve_artifact(
    root: Path,
    *,
    exp_id: str,
    model_name: str,
    dataset: str,
    seed: int,
) -> Path:
    if model_name == DEFAULT_HYBRID_SERVE_MODEL:
        return ensure_large_nano_hybrid_serve_artifact(
            root,
            exp_id=exp_id,
            model_name=model_name,
            dataset_id=dataset,
            seed=seed,
        )
    return ensure_large_nano_serve_artifact(
        root,
        exp_id=exp_id,
        model_name=model_name,
        dataset_id=dataset,
        seed=seed,
    )


def execute(dto: EvaluateServeModelDTO) -> Result[EvaluateServeModelResult, EvaluateServeModelError]:
    """Run real batch inference on a holdout split and return classification metrics."""
    init_correlation_id()

    try:
        open_dataset_feature_count(dto.dataset)
    except KeyError as exc:
        return fail(EvaluateServeModelError("UNSUPPORTED_DATASET", str(exc)))

    try:
        _ensure_serve_artifact(
            ROOT,
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
        )
        features, y_true = load_open_split_labeled(
            dto.dataset,
            ROOT,
            split=dto.split,
            n_rows=dto.n_rows,
            random_state=dto.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        return fail(EvaluateServeModelError("DATA_ERROR", str(exc)))

    if not features:
        return fail(EvaluateServeModelError("EMPTY_SPLIT", f"no rows in {dto.split} split"))

    batch_outcome = run_batch_predict(
        BatchPredictDTO(
            features=features,
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            chunk_size=dto.chunk_size,
        )
    )
    if isinstance(batch_outcome, Fail):
        return fail(EvaluateServeModelError(batch_outcome.error.code, batch_outcome.error.message))

    assert isinstance(batch_outcome, Ok)
    batch = batch_outcome.value
    probs = batch.probabilities
    y_pred = batch.labels

    roc_auc = float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else 0.5
    fpr, tpr, _ = roc_curve(y_true, probs)
    confusion = _confusion_stats(y_true, y_pred)

    sample_rows: list[dict[str, float | int]] = []
    for idx in range(min(10, len(y_true))):
        sample_rows.append(
            {
                "row": idx,
                "probability": round(probs[idx], 4),
                "predicted": y_pred[idx],
                "actual": y_true[idx],
                "correct": int(y_pred[idx] == y_true[idx]),
            }
        )

    result = EvaluateServeModelResult(
        exp_id=dto.exp_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        split=dto.split,
        n_rows=len(y_true),
        accuracy=float(accuracy_score(y_true, y_pred)),
        roc_auc=roc_auc,
        brier_score=float(brier_score_loss(y_true, probs)),
        confusion=confusion,
        checkpoint_path=batch.checkpoint_path,
        mean_probability=float(np.mean(probs)),
        positive_rate=float(np.mean(y_pred)),
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        sample_rows=sample_rows,
    )

    log_event(
        "info",
        "serve model evaluation",
        exp_id=dto.exp_id,
        model=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        split=dto.split,
        n_rows=result.n_rows,
        accuracy=round(result.accuracy, 4),
        roc_auc=round(result.roc_auc, 4),
        checkpoint_path=result.checkpoint_path,
        record_source="evaluate_serve_model",
    )
    return ok(result)
