"""Calibration evaluation for serve models on balanced holdout slices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.application.agro_validation_cases import AGRO_VALIDATION_CASES
from src.application.balanced_metrics import brier_score, expected_calibration_error
from src.application.batch_predict import BatchPredictDTO, run_batch_predict
from src.application.calibration import (
    CalibrationArtifact,
    apply_isotonic,
    fit_isotonic_calibrator,
    spearman_rank_correlation,
)
from src.application.clinical_validation_cases import CLINICAL_VALIDATION_CASES
from src.application.evaluate_serve_model import _ensure_serve_artifact, load_open_split_labeled
from src.application.human_agro_scorer import score_municipality
from src.application.human_cv_scorer import score_patient
from src.application.open_serve import DEFAULT_SERVE_MODEL, DEFAULT_SERVE_SEED
from src.shared.result import Fail, Ok, Result, fail, ok
from src.training.structured_log import init_correlation_id, log_event

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CV_DATASET = "synthea_cv_risk_v1"
DEFAULT_CV_EXP_ID = "exp_034"
RankingDomain = Literal["clinical", "agro"]


@dataclass(frozen=True)
class CalibrationEvaluationError:
    code: str
    message: str


@dataclass(frozen=True)
class CalibrationEvaluationDTO:
    exp_id: str = DEFAULT_CV_EXP_ID
    model_name: str = DEFAULT_SERVE_MODEL
    dataset: str = DEFAULT_CV_DATASET
    seed: int = DEFAULT_SERVE_SEED
    split: str = "val"
    n_rows: int = 1000
    min_negatives: int = 100
    fit_fraction: float = 0.8
    chunk_size: int = 2048
    force_balanced: bool = True
    ranking_domain: RankingDomain = "clinical"
    artifact_exp_id: str | None = None


@dataclass(frozen=True)
class CalibrationEvaluationResult:
    exp_id: str
    model_name: str
    dataset: str
    seed: int
    n_rows: int
    n_negatives: int
    n_positives: int
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    roc_auc_before: float
    roc_auc_after: float
    spearman_rho: float
    artifact_path: str
    passed: bool
    record_source: str = "calibration_evaluation"


def calibration_artifact_path(
    root: Path,
    *,
    exp_id: str,
    model_name: str,
    dataset: str,
    seed: int,
) -> Path:
    return (
        root
        / "artifacts"
        / exp_id
        / f"{model_name}_{dataset}"
        / f"seed_{seed}"
        / "calibration_isotonic.json"
    )


def save_calibration_artifact(path: Path, artifact: CalibrationArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")


def load_calibration_artifact(path: Path) -> CalibrationArtifact | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationArtifact.from_dict(payload)


def _roc_auc(labels: list[int], probs: list[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probs))


def _ranking_spearman(
    artifact: CalibrationArtifact,
    *,
    ranking_domain: RankingDomain,
    exp_id: str,
    model_name: str,
    dataset: str,
    seed: int,
) -> Result[float, CalibrationEvaluationError]:
    raw_probs: list[float] = []
    cal_probs: list[float] = []

    if ranking_domain == "agro":
        for case in AGRO_VALIDATION_CASES:
            scored = score_municipality(
                case.profile,
                exp_id=exp_id,
                model_name=model_name,
                dataset=dataset,
                seed=seed,
            )
            if isinstance(scored, Fail):
                return fail(CalibrationEvaluationError(scored.error.code, scored.error.message))
            assert isinstance(scored, Ok)
            raw = scored.value.probability
            raw_probs.append(raw)
            cal_probs.append(apply_isotonic([raw], artifact)[0])
    else:
        for case in CLINICAL_VALIDATION_CASES:
            scored = score_patient(case.profile)
            if isinstance(scored, Fail):
                return fail(CalibrationEvaluationError(scored.error.code, scored.error.message))
            assert isinstance(scored, Ok)
            raw = scored.value.probability
            raw_probs.append(raw)
            cal_probs.append(apply_isotonic([raw], artifact)[0])

    return ok(spearman_rank_correlation(raw_probs, cal_probs))


def run_calibration_evaluation(
    dto: CalibrationEvaluationDTO | None = None,
    *,
    min_spearman_rho: float = 0.85,
    max_ece_after: float = 0.08,
    require_ece_improved: bool = False,
    require_brier_improved: bool = False,
    min_auc_delta: float = -1.0,
) -> Result[CalibrationEvaluationResult, CalibrationEvaluationError]:
    init_correlation_id()
    dto = dto or CalibrationEvaluationDTO()
    artifact_exp = dto.artifact_exp_id or "exp_043"
    n_rows = None if dto.n_rows is None or dto.n_rows <= 0 else dto.n_rows

    try:
        _ensure_serve_artifact(
            ROOT,
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
        )
        features, labels = load_open_split_labeled(
            dto.dataset,
            ROOT,
            split=dto.split,
            n_rows=n_rows,
            random_state=dto.seed,
            min_negatives=dto.min_negatives,
            force_balanced=dto.force_balanced,
        )
    except (FileNotFoundError, ValueError) as exc:
        return fail(CalibrationEvaluationError("DATA_ERROR", str(exc)))

    if len(set(labels)) < 2:
        return fail(CalibrationEvaluationError("SINGLE_CLASS", "balanced slice has one class"))

    fit_idx, eval_idx = train_test_split(
        list(range(len(labels))),
        train_size=dto.fit_fraction,
        stratify=labels,
        random_state=dto.seed,
    )
    fit_features = [features[i] for i in fit_idx]
    eval_features = [features[i] for i in eval_idx]
    fit_labels = [labels[i] for i in fit_idx]
    eval_labels = [labels[i] for i in eval_idx]

    fit_batch = run_batch_predict(
        BatchPredictDTO(
            features=fit_features,
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            chunk_size=dto.chunk_size,
        )
    )
    if isinstance(fit_batch, Fail):
        return fail(CalibrationEvaluationError(fit_batch.error.code, fit_batch.error.message))

    eval_batch = run_batch_predict(
        BatchPredictDTO(
            features=eval_features,
            exp_id=dto.exp_id,
            model_name=dto.model_name,
            dataset=dto.dataset,
            seed=dto.seed,
            chunk_size=dto.chunk_size,
        )
    )
    if isinstance(eval_batch, Fail):
        return fail(CalibrationEvaluationError(eval_batch.error.code, eval_batch.error.message))

    assert isinstance(fit_batch, Ok) and isinstance(eval_batch, Ok)
    fit_probs = fit_batch.value.probabilities
    eval_probs = eval_batch.value.probabilities

    artifact = fit_isotonic_calibrator(fit_probs, fit_labels)
    calibrated_eval = apply_isotonic(eval_probs, artifact)

    ece_before = expected_calibration_error(eval_labels, eval_probs)
    ece_after = expected_calibration_error(eval_labels, calibrated_eval)
    brier_before = brier_score(eval_labels, eval_probs)
    brier_after = brier_score(eval_labels, calibrated_eval)
    auc_before = _roc_auc(eval_labels, eval_probs)
    auc_after = _roc_auc(eval_labels, calibrated_eval)

    spearman_outcome = _ranking_spearman(
        artifact,
        ranking_domain=dto.ranking_domain,
        exp_id=dto.exp_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
    )
    if isinstance(spearman_outcome, Fail):
        return spearman_outcome
    assert isinstance(spearman_outcome, Ok)
    spearman = spearman_outcome.value

    artifact_path = calibration_artifact_path(
        ROOT,
        exp_id=artifact_exp,
        model_name=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
    )
    save_calibration_artifact(artifact_path, artifact)

    n_neg = sum(1 for y in labels if y == 0)
    passed = ece_after <= max_ece_after and spearman >= min_spearman_rho
    if require_ece_improved:
        passed = passed and ece_after < ece_before
    if require_brier_improved:
        passed = passed and brier_after <= brier_before + 1e-9
    if (auc_after - auc_before) < min_auc_delta:
        passed = False

    result = CalibrationEvaluationResult(
        exp_id=dto.exp_id,
        model_name=dto.model_name,
        dataset=dto.dataset,
        seed=dto.seed,
        n_rows=len(labels),
        n_negatives=n_neg,
        n_positives=len(labels) - n_neg,
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_before,
        brier_after=brier_after,
        roc_auc_before=auc_before,
        roc_auc_after=auc_after,
        spearman_rho=spearman,
        artifact_path=str(artifact_path),
        passed=passed,
    )
    log_event(
        "info",
        "calibration evaluation complete",
        exp_id=dto.exp_id,
        artifact_exp_id=artifact_exp,
        ranking_domain=dto.ranking_domain,
        ece_before=round(ece_before, 4),
        ece_after=round(ece_after, 4),
        brier_before=round(brier_before, 4),
        brier_after=round(brier_after, 4),
        spearman_rho=round(spearman, 4),
        passed=passed,
        record_source="calibration_evaluation",
    )
    return ok(result)
