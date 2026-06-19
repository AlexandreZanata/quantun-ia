"""Build a single consolidated report with all serve model and human-demo results."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.application.evaluate_serve_model import (
    EvaluateServeModelDTO,
    list_serve_models,
)
from src.application.evaluate_serve_model import (
    execute as evaluate_execute,
)
from src.application.human_cv_scorer import PatientProfile, compare_profiles, score_patient
from src.shared.result import Fail, Ok

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "logs" / "model_results_summary.json"

HUMAN_SCENARIOS: tuple[tuple[str, PatientProfile, PatientProfile], ...] = (
    (
        "young_healthy_vs_elderly_smoker",
        PatientProfile(age_years=32, sex_male=True, bmi=24, systolic_bp=118, smoker=False),
        PatientProfile(
            age_years=78, sex_male=True, bmi=31, systolic_bp=165,
            smoker=True, diabetes=True, prior_mi=True,
        ),
    ),
    (
        "same_age_lifestyle_contrast",
        PatientProfile(age_years=55, sex_male=False, bmi=23, systolic_bp=120, hdl=65, smoker=False),
        PatientProfile(
            age_years=55, sex_male=True, bmi=34, systolic_bp=152,
            hdl=38, smoker=True, diabetes=True,
        ),
    ),
    (
        "baseline_patient",
        PatientProfile(age_years=55, sex_male=False, bmi=28, systolic_bp=128),
        PatientProfile(age_years=55, sex_male=False, bmi=28, systolic_bp=128),
    ),
)


def _load_benchmark_holdout_rows() -> list[dict[str, Any]]:
    from dashboard.benchmark_data import latest_holdout_records, load_records, to_leaderboard_rows

    records = load_records()
    holdout = latest_holdout_records(records)
    rows = to_leaderboard_rows(holdout if holdout else records)
    return [
        {
            "exp_id": row.get("exp_id"),
            "model": row.get("model"),
            "accuracy_pct": row.get("accuracy"),
            "loss": row.get("loss"),
            "elapsed_s": row.get("elapsed_s"),
            "eval_set": row.get("eval_set"),
            "source": row.get("source"),
        }
        for row in rows
    ]


def _evaluate_result_to_dict(result) -> dict[str, Any]:
    return {
        "exp_id": result.exp_id,
        "model_name": result.model_name,
        "dataset": result.dataset,
        "seed": result.seed,
        "split": result.split,
        "n_rows": result.n_rows,
        "roc_auc": round(result.roc_auc, 4),
        "accuracy": round(result.accuracy, 4),
        "accuracy_pct": round(result.accuracy * 100.0, 2),
        "brier_score": round(result.brier_score, 4),
        "mean_probability": round(result.mean_probability, 4),
        "positive_rate": round(result.positive_rate, 4),
        "confusion": {
            "true_negative": result.confusion.true_negative,
            "false_positive": result.confusion.false_positive,
            "false_negative": result.confusion.false_negative,
            "true_positive": result.confusion.true_positive,
        },
        "checkpoint_path": result.checkpoint_path,
        "sample_predictions": result.sample_rows[:5],
    }


def _score_to_dict(profile: PatientProfile, result) -> dict[str, Any]:
    return {
        "summary": result.human_summary,
        "risk_percent": round(result.risk_percent, 2),
        "risk_band": result.risk_band,
        "probability": round(result.probability, 4),
        "profile": {
            "age_years": profile.age_years,
            "sex_male": profile.sex_male,
            "bmi": profile.bmi,
            "systolic_bp": profile.systolic_bp,
            "smoker": profile.smoker,
            "diabetes": profile.diabetes,
            "prior_mi": profile.prior_mi,
        },
    }


def build_model_results_report(
    *,
    n_rows: int = 5000,
    split: str = "val",
    chunk_size: int = 2048,
    include_benchmark: bool = True,
) -> dict[str, Any]:
    """Evaluate all serve checkpoints and human clinic scenarios into one dict."""
    serve_results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for model in list_serve_models():
        outcome = evaluate_execute(
            EvaluateServeModelDTO(
                exp_id=model.exp_id,
                model_name=model.model_name,
                dataset=model.dataset,
                seed=model.seed,
                split=split,
                n_rows=n_rows,
                chunk_size=chunk_size,
            )
        )
        if isinstance(outcome, Fail):
            errors.append(
                {
                    "label": model.label,
                    "exp_id": model.exp_id,
                    "code": outcome.error.code,
                    "message": outcome.error.message,
                }
            )
            continue
        assert isinstance(outcome, Ok)
        row = _evaluate_result_to_dict(outcome.value)
        row["label"] = model.label
        serve_results.append(row)

    human_scenarios: list[dict[str, Any]] = []
    for scenario_id, profile_a, profile_b in HUMAN_SCENARIOS:
        if profile_a == profile_b:
            outcome = score_patient(profile_a)
            if isinstance(outcome, Fail):
                errors.append({"scenario": scenario_id, "code": outcome.error.code, "message": outcome.error.message})
                continue
            assert isinstance(outcome, Ok)
            human_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "type": "single_patient",
                    "patient": _score_to_dict(profile_a, outcome.value),
                }
            )
        else:
            outcome = compare_profiles(profile_a, profile_b)
            if isinstance(outcome, Fail):
                errors.append({"scenario": scenario_id, "code": outcome.error.code, "message": outcome.error.message})
                continue
            assert isinstance(outcome, Ok)
            ra, rb = outcome.value
            higher = "A" if ra.probability > rb.probability else "B" if rb.probability > ra.probability else "equal"
            human_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "type": "comparison",
                    "patient_a": _score_to_dict(profile_a, ra),
                    "patient_b": _score_to_dict(profile_b, rb),
                    "higher_risk": higher,
                    "delta_risk_pp": round(abs(ra.risk_percent - rb.risk_percent), 2),
                }
            )

    import torch

    report: dict[str, Any] = {
        "report_type": "model_results_summary",
        "version": "1.0",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "hardware": {
            "qml_device": os.environ.get("QML_DEVICE", "cpu"),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "evaluation_config": {
            "split": split,
            "n_rows_per_model": n_rows,
            "chunk_size": chunk_size,
        },
        "serve_models": serve_results,
        "human_clinic_scenarios": human_scenarios,
        "errors": errors,
    }

    if include_benchmark:
        report["benchmark_holdout"] = _load_benchmark_holdout_rows()

    return report


def write_model_results_report(
    path: Path | None = None,
    *,
    n_rows: int = 5000,
    split: str = "val",
) -> Path:
    """Write consolidated JSON report to logs/model_results_summary.json."""
    out = path or DEFAULT_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_model_results_report(n_rows=n_rows, split=split)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out


def load_model_results_report(path: Path | None = None) -> dict[str, Any] | None:
    """Load report if present."""
    out = path or DEFAULT_REPORT_PATH
    if not out.is_file():
        return None
    return json.loads(out.read_text(encoding="utf-8"))
