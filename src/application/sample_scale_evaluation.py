"""Evaluate serve-model metrics at increasing holdout sample sizes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.application.batch_predict import BatchPredictDTO, run_batch_predict
from src.application.evaluate_serve_model import (
    _confusion_stats,
    _ensure_serve_artifact,
    load_open_split_labeled,
)
from src.application.open_serve import (
    DEFAULT_SERVE_EXP_ID,
    DEFAULT_SERVE_MODEL,
    DEFAULT_SERVE_SEED,
)
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.structured_log import init_correlation_id, log_event

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CV_DATASET = "synthea_cv_risk_v1"
DEFAULT_CV_EXP_ID = "exp_034"

SAMPLE_SCALE_SIZES: tuple[int, ...] = tuple(range(100, 2001, 100))


@dataclass(frozen=True)
class SampleScaleEvaluationError:
    code: str
    message: str


@dataclass(frozen=True)
class SampleScalePoint:
    n_rows: int
    n_negatives: int
    n_positives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier_score: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class SampleScaleCurveResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    split: str
    points: tuple[SampleScalePoint, ...]
    record_source: str = "sample_scale_evaluation"


@dataclass(frozen=True)
class HoldoutPredictionRow:
    row_index: int
    probability: float
    predicted: int
    actual: int
    correct: int


@dataclass(frozen=True)
class HoldoutPredictionsExport:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    split: str
    n_rows: int
    n_negatives: int
    n_positives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    rows: tuple[HoldoutPredictionRow, ...]
    record_source: str = "holdout_predictions_export"


@dataclass(frozen=True)
class SampleScaleEvaluationDTO:
    exp_id: str = DEFAULT_CV_EXP_ID
    model_name: str = DEFAULT_SERVE_MODEL
    dataset: str = DEFAULT_CV_DATASET
    seed: int = DEFAULT_SERVE_SEED
    split: str = "val"
    sample_sizes: tuple[int, ...] = SAMPLE_SCALE_SIZES
    chunk_size: int = 2048


@dataclass(frozen=True)
class HoldoutPredictionsDTO:
    exp_id: str = DEFAULT_CV_EXP_ID
    model_name: str = DEFAULT_SERVE_MODEL
    dataset: str = DEFAULT_CV_DATASET
    seed: int = DEFAULT_SERVE_SEED
    split: str = "val"
    n_rows: int = 100
    chunk_size: int = 2048


def _classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    probs: list[float],
) -> tuple[float, float, float, float, float, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else 0.5
    brier = float(brier_score_loss(y_true, probs))
    return accuracy, precision, recall, f1, roc_auc, brier


def _predict_holdout(
    dto: SampleScaleEvaluationDTO | HoldoutPredictionsDTO,
    *,
    n_rows: int,
) -> Result[tuple[list[int], list[int], list[float], str], SampleScaleEvaluationError]:
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
            n_rows=n_rows,
            random_state=dto.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        return fail(SampleScaleEvaluationError("DATA_ERROR", str(exc)))

    if not features:
        return fail(SampleScaleEvaluationError("EMPTY_SPLIT", f"no rows in {dto.split} split"))

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
        return fail(SampleScaleEvaluationError(batch_outcome.error.code, batch_outcome.error.message))

    assert isinstance(batch_outcome, Ok)
    batch = batch_outcome.value
    return ok((y_true, batch.labels, batch.probabilities, batch.checkpoint_path))


def run_sample_scale_curve(
    dto: SampleScaleEvaluationDTO | None = None,
) -> Result[SampleScaleCurveResult, SampleScaleEvaluationError]:
    """Evaluate metrics at n=100, 200, …, 2000 (stratified subsamples)."""
    init_correlation_id()
    dto = dto or SampleScaleEvaluationDTO()

    points: list[SampleScalePoint] = []
    for n_rows in dto.sample_sizes:
        outcome = _predict_holdout(dto, n_rows=n_rows)
        if isinstance(outcome, Fail):
            return outcome

        y_true, y_pred, probs, _ = outcome.value
        accuracy, precision, recall, f1, roc_auc, brier = _classification_metrics(
            y_true, y_pred, probs
        )
        confusion = _confusion_stats(y_true, y_pred)
        n_neg = sum(1 for y in y_true if y == 0)
        n_pos = len(y_true) - n_neg
        points.append(
            SampleScalePoint(
                n_rows=n_rows,
                n_negatives=n_neg,
                n_positives=n_pos,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                roc_auc=roc_auc,
                brier_score=brier,
                true_positive=confusion.true_positive,
                true_negative=confusion.true_negative,
                false_positive=confusion.false_positive,
                false_negative=confusion.false_negative,
            )
        )

    result = SampleScaleCurveResult(
        exp_id=dto.exp_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        split=dto.split,
        points=tuple(points),
    )
    log_event(
        "info",
        "sample scale curve complete",
        exp_id=dto.exp_id,
        dataset=dto.dataset,
        n_points=len(points),
        record_source="sample_scale_evaluation",
    )
    return ok(result)


def export_holdout_predictions(
    dto: HoldoutPredictionsDTO | None = None,
) -> Result[HoldoutPredictionsExport, SampleScaleEvaluationError]:
    """Score n holdout rows and return per-row predictions with aggregate metrics."""
    init_correlation_id()
    dto = dto or HoldoutPredictionsDTO()

    outcome = _predict_holdout(dto, n_rows=dto.n_rows)
    if isinstance(outcome, Fail):
        return outcome

    y_true, y_pred, probs, _ = outcome.value
    accuracy, precision, recall, f1, roc_auc, _ = _classification_metrics(
        y_true, y_pred, probs
    )
    n_neg = sum(1 for y in y_true if y == 0)
    n_pos = len(y_true) - n_neg
    rows = tuple(
        HoldoutPredictionRow(
            row_index=idx,
            probability=round(probs[idx], 6),
            predicted=y_pred[idx],
            actual=y_true[idx],
            correct=int(y_pred[idx] == y_true[idx]),
        )
        for idx in range(len(y_true))
    )
    return ok(
        HoldoutPredictionsExport(
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            split=dto.split,
            n_rows=len(rows),
            n_negatives=n_neg,
            n_positives=n_pos,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            rows=rows,
        )
    )


def curve_to_dict(result: SampleScaleCurveResult) -> dict:
    return {
        "exp_id": result.exp_id,
        "model_name": result.model_name,
        "dataset": result.dataset,
        "seed": result.seed,
        "split": result.split,
        "record_source": result.record_source,
        "points": [
            {
                "n_rows": p.n_rows,
                "n_negatives": p.n_negatives,
                "n_positives": p.n_positives,
                "accuracy": round(p.accuracy, 6),
                "precision": round(p.precision, 6),
                "recall": round(p.recall, 6),
                "f1": round(p.f1, 6),
                "roc_auc": round(p.roc_auc, 6),
                "brier_score": round(p.brier_score, 6),
                "confusion": {
                    "tp": p.true_positive,
                    "tn": p.true_negative,
                    "fp": p.false_positive,
                    "fn": p.false_negative,
                },
            }
            for p in result.points
        ],
    }


def predictions_to_dict(result: HoldoutPredictionsExport) -> dict:
    return {
        "exp_id": result.exp_id,
        "model_name": result.model_name,
        "dataset": result.dataset,
        "seed": result.seed,
        "split": result.split,
        "n_rows": result.n_rows,
        "n_negatives": result.n_negatives,
        "n_positives": result.n_positives,
        "accuracy": round(result.accuracy, 6),
        "precision": round(result.precision, 6),
        "recall": round(result.recall, 6),
        "f1": round(result.f1, 6),
        "roc_auc": round(result.roc_auc, 6),
        "record_source": result.record_source,
        "rows": [
            {
                "row_index": r.row_index,
                "probability": r.probability,
                "predicted": r.predicted,
                "actual": r.actual,
                "correct": r.correct,
            }
            for r in result.rows
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
