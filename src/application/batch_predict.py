"""Batch inference pipeline — CSV/JSON in, probabilities out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.application.dto import PredictNanomodelDTO
from src.application.open_serve import open_dataset_feature_count
from src.application.predict_nanomodel import execute as predict_execute
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.champion import load_champion_manifest
from src.training.checkpoints import resolve_checkpoint_dir
from src.training.structured_log import init_correlation_id, log_event

BREAST_CANCER_FEATURE_COUNT = 30
DEFAULT_EXP_ID = "quantum_nano_bc_app"
DEFAULT_MODEL = "hybrid_sandwich"
DEFAULT_DATASET = "breast_cancer"
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 64

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


@dataclass(frozen=True)
class BatchPredictError:
    code: str
    message: str


@dataclass(frozen=True)
class BatchPredictDTO:
    features: list[list[float]]
    exp_id: str = DEFAULT_EXP_ID
    model_name: str = DEFAULT_MODEL
    dataset: str = DEFAULT_DATASET
    seed: int = DEFAULT_SEED
    chunk_size: int = DEFAULT_CHUNK_SIZE


@dataclass(frozen=True)
class BatchPredictResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    probabilities: list[float]
    labels: list[int]
    checkpoint_path: str
    n_rows: int
    record_source: str = "batch_predict"


def breast_cancer_column_names() -> list[str]:
    return list(load_breast_cancer().feature_names)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    if all(name in frame.columns for name in breast_cancer_column_names()):
        return breast_cancer_column_names()
    feature_cols = [c for c in frame.columns if c.startswith("feature_")]
    if feature_cols:
        return sorted(feature_cols, key=lambda c: int(c.split("_", 1)[1]))
    numeric = [c for c in frame.columns if c != "label"]
    return numeric


def load_input_rows(path: Path, *, dataset: str = DEFAULT_DATASET) -> tuple[list[list[float]], list[str]]:
    """Load raw feature rows from CSV or JSON."""
    expected_features = (
        BREAST_CANCER_FEATURE_COUNT
        if dataset == DEFAULT_DATASET
        else open_dataset_feature_count(dataset)
    )
    if not path.is_file():
        raise FileNotFoundError(f"input not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "features" in payload:
            rows = [[float(v) for v in row] for row in payload["features"]]
        elif isinstance(payload, list):
            rows = [[float(v) for v in row] for row in payload]
        else:
            raise ValueError('JSON must be a list of rows or {"features": [...]}')
        columns = (
            breast_cancer_column_names()
            if dataset == DEFAULT_DATASET
            else [f"feature_{i}" for i in range(expected_features)]
        )
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        columns = _feature_columns(frame)
        rows = frame[columns].astype(float).values.tolist()
    else:
        raise ValueError(f"unsupported input format: {suffix}")

    if not rows:
        raise ValueError("input contains no feature rows")
    expected = len(columns) if columns else expected_features
    if expected != expected_features:
        raise ValueError(f"expected {expected_features} features, got {expected}")
    for idx, row in enumerate(rows):
        if len(row) != expected_features:
            raise ValueError(
                f"row {idx} must have {expected_features} features (got {len(row)})"
            )
    return rows, columns


def max_probability_delta(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("probability lists must have equal length")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def run_batch_predict(dto: BatchPredictDTO) -> Result[BatchPredictResult, BatchPredictError]:
    """Score rows in chunks via predict_nanomodel (same path as API)."""
    init_correlation_id()
    if not dto.features:
        return fail(BatchPredictError("INVALID_FEATURES", "features must not be empty"))

    err = _validate_rows(dto.features, dto.dataset)
    if err is not None:
        return fail(err)

    probabilities: list[float] = []
    labels: list[int] = []
    checkpoint_path = ""

    for start in range(0, len(dto.features), dto.chunk_size):
        chunk = dto.features[start : start + dto.chunk_size]
        outcome = predict_execute(
            PredictNanomodelDTO(
                exp_id=dto.exp_id,
                model_name=dto.model_name,
                dataset=dto.dataset,
                seed=dto.seed,
                features=chunk,
            )
        )
        if isinstance(outcome, Fail):
            return fail(BatchPredictError(outcome.error.code, outcome.error.message))
        assert isinstance(outcome, Ok)
        result = outcome.value
        probabilities.extend(result.probabilities)
        labels.extend(result.labels)
        checkpoint_path = result.checkpoint_path

    log_event(
        "info",
        "batch predict complete",
        exp_id=dto.exp_id,
        model=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        n_rows=len(dto.features),
        checkpoint_path=checkpoint_path,
        record_source="batch_predict",
    )
    return ok(
        BatchPredictResult(
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            probabilities=probabilities,
            labels=labels,
            checkpoint_path=checkpoint_path,
            n_rows=len(dto.features),
        )
    )


def _validate_rows(rows: list[list[float]], dataset: str) -> BatchPredictError | None:
    expected = (
        BREAST_CANCER_FEATURE_COUNT
        if dataset == DEFAULT_DATASET
        else open_dataset_feature_count(dataset)
    )
    for idx, row in enumerate(rows):
        if len(row) != expected:
            return BatchPredictError(
                "INVALID_FEATURES",
                f"row {idx} must have {expected} features",
            )
    return None


def _checkpoint_mtime(exp_id: str, model_name: str, dataset: str, seed: int) -> str:
    directory = resolve_checkpoint_dir(exp_id, model_name, dataset, seed=seed)
    best = directory / "best.pt"
    if best.is_file():
        return datetime.fromtimestamp(best.stat().st_mtime, tz=timezone.utc).isoformat()
    manifest = load_champion_manifest()
    if manifest is not None:
        return manifest.promoted_at
    return "unknown"


def build_output_header(
    result: BatchPredictResult,
    *,
    source_input: str | None = None,
) -> list[str]:
    mtime = _checkpoint_mtime(result.exp_id, result.model_name, result.dataset, result.seed)
    lines = [
        f"# exp_id={result.exp_id}",
        f"# model_name={result.model_name}",
        f"# dataset={result.dataset}",
        f"# seed={result.seed}",
        f"# checkpoint_path={result.checkpoint_path}",
        f"# checkpoint_mtime={mtime}",
        f"# n_rows={result.n_rows}",
        f"# generated_at={datetime.now(timezone.utc).isoformat()}",
    ]
    if source_input:
        lines.append(f"# source_input={source_input}")
    return lines


def write_output_csv(
    path: Path,
    result: BatchPredictResult,
    *,
    source_input: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = build_output_header(result, source_input=source_input)
    frame = pd.DataFrame(
        {
            "row": list(range(result.n_rows)),
            "probability": result.probabilities,
            "label": result.labels,
        }
    )
    body = frame.to_csv(index=False)
    path.write_text("\n".join(header_lines) + "\n" + body, encoding="utf-8")
    return path


def write_output_json(
    path: Path,
    result: BatchPredictResult,
    *,
    source_input: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "exp_id": result.exp_id,
        "model_name": result.model_name,
        "dataset": result.dataset,
        "seed": result.seed,
        "checkpoint_path": result.checkpoint_path,
        "checkpoint_mtime": _checkpoint_mtime(
            result.exp_id, result.model_name, result.dataset, result.seed
        ),
        "n_rows": result.n_rows,
        "probabilities": result.probabilities,
        "labels": result.labels,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_input:
        payload["source_input"] = source_input
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def predict_request_payload_all_rows(dto: BatchPredictDTO) -> dict[str, Any]:
    return {
        "exp_id": dto.exp_id,
        "model_name": dto.model_name,
        "dataset": dto.dataset,
        "seed": dto.seed,
        "features": dto.features,
    }
