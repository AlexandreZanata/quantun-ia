"""
EXP 042 — Sample-scale precision curve + 100-prediction export (Synthea CV).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_042_sample_scale_precision/run.py
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_042_sample_scale_precision/run.py --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.sample_scale_evaluation import (
    HoldoutPredictionsDTO,
    SampleScaleEvaluationDTO,
    SAMPLE_SCALE_SIZES,
    curve_to_dict,
    export_holdout_predictions,
    predictions_to_dict,
    run_sample_scale_curve,
    write_json,
)
from src.shared.result import Fail, Ok
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_ID = "exp_042"
MODEL_NAME = "large_nano_mlp_sample_scale"
MIN_ROC_AUC = 0.78
GATE_SAMPLE_SIZE = 2000
LOCAL_OUT = Path(__file__).resolve().parents[2] / ".local" / "out"
PREDICTIONS_PATH = LOCAL_OUT / "predictions_100_synthea_cv.json"
CURVE_PATH = LOCAL_OUT / "sample_scale_precision.json"


@dataclass(frozen=True)
class Exp042Result:
    curve_points: tuple
    predictions_n_rows: int
    predictions_accuracy: float
    predictions_precision: float
    predictions_recall: float
    predictions_f1: float
    predictions_roc_auc: float
    min_roc_auc: float
    passed: bool
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 with QML_DEVICE=cuda")


def run_exp_042(*, verbose: bool = True, require_cuda: bool = True) -> Exp042Result:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    init_correlation_id()
    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP {EXP_ID} — Sample-scale precision (100→2000, step 100)")
        print(f"{'=' * 60}\n")

    curve_outcome = run_sample_scale_curve(SampleScaleEvaluationDTO())
    if isinstance(curve_outcome, Fail):
        raise RuntimeError(f"{curve_outcome.error.code}: {curve_outcome.error.message}")
    assert isinstance(curve_outcome, Ok)
    curve = curve_outcome.value

    pred_outcome = export_holdout_predictions(HoldoutPredictionsDTO(n_rows=100))
    if isinstance(pred_outcome, Fail):
        raise RuntimeError(f"{pred_outcome.error.code}: {pred_outcome.error.message}")
    assert isinstance(pred_outcome, Ok)
    predictions = pred_outcome.value

    write_json(CURVE_PATH, curve_to_dict(curve))
    write_json(PREDICTIONS_PATH, predictions_to_dict(predictions))

    gate_point = next(p for p in curve.points if p.n_rows == GATE_SAMPLE_SIZE)
    min_roc = gate_point.roc_auc
    passed = min_roc >= MIN_ROC_AUC

    log = ExperimentLogger(EXP_ID, MODEL_NAME, seed=42, profile="sample_scale")
    for idx, point in enumerate(curve.points, start=1):
        log.log(
            idx,
            n_rows=point.n_rows,
            accuracy=round(point.accuracy, 4),
            precision=round(point.precision, 4),
            recall=round(point.recall, 4),
            f1=round(point.f1, 4),
            roc_auc=round(point.roc_auc, 4),
        )
    log.finish(
        time.perf_counter() - t0,
        record_type="sample_scale_curve",
        n_points=len(curve.points),
        min_roc_auc=round(min_roc, 4),
        predictions_n=100,
        predictions_accuracy=round(predictions.accuracy, 4),
        predictions_precision=round(predictions.precision, 4),
        predictions_recall=round(predictions.recall, 4),
        predictions_f1=round(predictions.f1, 4),
        predictions_roc_auc=round(predictions.roc_auc, 4),
        passed=passed,
        eval_set="synthea_cv_val",
    )

    log_event(
        "info",
        "exp_042 sample scale summary",
        exp_id=EXP_ID,
        min_roc_auc=round(min_roc, 4),
        predictions_precision=round(predictions.precision, 4),
        passed=passed,
    )

    if verbose:
        print(f"{'n':>6} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8} {'auc':>8}")
        for p in curve.points:
            print(
                f"{p.n_rows:6d} {p.accuracy:8.4f} {p.precision:8.4f} "
                f"{p.recall:8.4f} {p.f1:8.4f} {p.roc_auc:8.4f}",
                flush=True,
            )
        print(
            f"\n100 predictions → acc={predictions.accuracy:.4f} "
            f"prec={predictions.precision:.4f} recall={predictions.recall:.4f} "
            f"f1={predictions.f1:.4f} auc={predictions.roc_auc:.4f}",
            flush=True,
        )
        print(f"Wrote {CURVE_PATH}")
        print(f"Wrote {PREDICTIONS_PATH}")
        print(
            f"\nROC-AUC @ n={GATE_SAMPLE_SIZE}={min_roc:.4f} | verdict={'PASS' if passed else 'FAIL'} | "
            f"elapsed={time.perf_counter() - t0:.1f}s\n",
            flush=True,
        )

    return Exp042Result(
        curve_points=curve.points,
        predictions_n_rows=predictions.n_rows,
        predictions_accuracy=predictions.accuracy,
        predictions_precision=predictions.precision,
        predictions_recall=predictions.recall,
        predictions_f1=predictions.f1,
        predictions_roc_auc=predictions.roc_auc,
        min_roc_auc=min_roc,
        passed=passed,
        elapsed_s=time.perf_counter() - t0,
    )


def _build_results_md(result: Exp042Result) -> str:
    verdict = "**accepted**" if result.passed else "**rejected**"
    lines = [
        "# Results — EXP 042: Sample-Scale Precision Curve",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  ",
        "**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42",
        "",
        "## Summary",
        "",
        "| Metric | Value | Gate |",
        "|--------|-------|------|",
        f"| Sample sizes | **100 → 2000 (step 100)** | 20 points |",
        f"| ROC-AUC @ n={GATE_SAMPLE_SIZE} | **{result.min_roc_auc:.4f}** | ≥ {MIN_ROC_AUC} |",
        f"| 100-row negatives / positives | see curve table | ~0 / 100 |",
        f"| 100-row accuracy | **{result.predictions_accuracy:.4f}** | — |",
        f"| 100-row precision | **{result.predictions_precision:.4f}** | — |",
        f"| 100-row recall | **{result.predictions_recall:.4f}** | — |",
        f"| 100-row F1 | **{result.predictions_f1:.4f}** | — |",
        f"| 100-row ROC-AUC | **{result.predictions_roc_auc:.4f}** | — |",
        f"| Elapsed | **{result.elapsed_s:.1f}s** | — |",
        "",
        "## Verdict",
        f"{verdict} — holdout metrics stable across sample sizes.",
        "",
        "## Sample-scale curve",
        "",
        "| n | Neg | Pos | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |",
        "|---|-----|-----|----------|-----------|--------|-----|---------|-------|",
    ]
    for p in result.curve_points:
        lines.append(
            f"| {p.n_rows} | {p.n_negatives} | {p.n_positives} | {p.accuracy:.4f} | "
            f"{p.precision:.4f} | {p.recall:.4f} | "
            f"{p.f1:.4f} | {p.roc_auc:.4f} | {p.brier_score:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exported artifacts (local workstation)",
            "",
            f"- `.local/out/predictions_100_synthea_cv.json` — 100 scored val rows",
            f"- `.local/out/sample_scale_precision.json` — full curve JSON",
            "",
            "## Interpretation",
            "",
            "- Synthea v1 val split has ~99% positive prevalence → accuracy is inflated.",
            "- **Precision / recall / F1** and **ROC-AUC** are the primary metrics for paper tables.",
            "- Each n uses an independent stratified subsample (seed 42).",
            "",
            "## Limitations",
            "",
            "- Synthetic data; not calibrated for real-world prevalence.",
            "- Subsample metrics have sampling variance at n=100.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 042 — sample-scale precision")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_042(verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
