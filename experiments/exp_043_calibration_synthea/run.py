"""
EXP 043 — Isotonic calibration on balanced Synthea CV holdout.

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_043_calibration_synthea/run.py
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_043_calibration_synthea/run.py --write-results
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

from src.application.calibration_evaluation import (
    CalibrationEvaluationDTO,
    run_calibration_evaluation,
)
from src.shared.result import Fail, Ok
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_ID = "exp_043"
MODEL_NAME = "large_nano_mlp_calibration"
MAX_ECE_AFTER = 0.085
MIN_SPEARMAN = 0.85


@dataclass(frozen=True)
class Exp043Result:
    n_rows: int
    n_negatives: int
    ece_before: float
    ece_after: float
    spearman_rho: float
    artifact_path: str
    passed: bool
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 with QML_DEVICE=cuda")


def run_exp_043(*, verbose: bool = True, require_cuda: bool = True) -> Exp043Result:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    init_correlation_id()
    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP {EXP_ID} — Isotonic calibration (balanced val slice)")
        print(f"{'=' * 60}\n")

    outcome = run_calibration_evaluation(
        CalibrationEvaluationDTO(),
        min_spearman_rho=MIN_SPEARMAN,
        max_ece_after=MAX_ECE_AFTER,
    )
    if isinstance(outcome, Fail):
        raise RuntimeError(f"{outcome.error.code}: {outcome.error.message}")
    assert isinstance(outcome, Ok)
    result = outcome.value

    log = ExperimentLogger(EXP_ID, MODEL_NAME, seed=42, profile="calibration")
    log.log(
        1,
        ece_before=round(result.ece_before, 4),
        ece_after=round(result.ece_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
    )
    log.finish(
        time.perf_counter() - t0,
        record_type="calibration_evaluation",
        n_rows=result.n_rows,
        n_negatives=result.n_negatives,
        ece_before=round(result.ece_before, 4),
        ece_after=round(result.ece_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
        artifact_path=result.artifact_path,
        passed=result.passed,
        eval_set="synthea_cv_val_balanced",
    )

    log_event(
        "info",
        "exp_043 calibration summary",
        exp_id=EXP_ID,
        ece_after=round(result.ece_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
        passed=result.passed,
    )

    if verbose:
        print(
            f"balanced n={result.n_rows} (neg={result.n_negatives}) | "
            f"ECE {result.ece_before:.4f} → {result.ece_after:.4f} | "
            f"Spearman ρ={result.spearman_rho:.4f}",
            flush=True,
        )
        print(f"artifact → {result.artifact_path}")
        print(
            f"verdict={'PASS' if result.passed else 'FAIL'} | elapsed={time.perf_counter() - t0:.1f}s\n",
            flush=True,
        )

    return Exp043Result(
        n_rows=result.n_rows,
        n_negatives=result.n_negatives,
        ece_before=result.ece_before,
        ece_after=result.ece_after,
        spearman_rho=result.spearman_rho,
        artifact_path=result.artifact_path,
        passed=result.passed,
        elapsed_s=time.perf_counter() - t0,
    )


def _build_results_md(result: Exp043Result) -> str:
    verdict = "**accepted**" if result.passed else "**rejected**"
    return "\n".join(
        [
            "# Results — EXP 043: Isotonic Calibration (Synthea CV)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  ",
            "**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42",
            "",
            "## Summary",
            "",
            "| Metric | Value | Gate |",
            "|--------|-------|------|",
            f"| Balanced rows | **{result.n_rows}** | ≥ 500 |",
            f"| Negatives | **{result.n_negatives}** | ≥ 50 |",
            f"| ECE before | **{result.ece_before:.4f}** | — |",
            f"| ECE after | **{result.ece_after:.4f}** | ≤ {MAX_ECE_AFTER} |",
            f"| Spearman ρ (clinical) | **{result.spearman_rho:.4f}** | ≥ {MIN_SPEARMAN} |",
            f"| Elapsed | **{result.elapsed_s:.1f}s** | — |",
            "",
            "## Verdict",
            f"{verdict} — isotonic calibration preserves ranking while improving ECE.",
            "",
            "## Artifact",
            "",
            f"- `{result.artifact_path}`",
            "",
            "## Limitations",
            "",
            "- Synthetic Synthea cohort; calibrated % not for clinical use.",
            "- Isotonic fit on balanced research slice only.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 043 — isotonic calibration")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_043(verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
