"""
EXP 082 — Isotonic calibration on ACYD C4 LargeNanoMLP (agro ranking).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_082_calibration_acyd/run.py --profile publication --write-results
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
from src.training.config import load_experiment_config
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_082_calibration_acyd"
EXP_ID = "exp_082"
MODEL_NAME = "large_nano_mlp_calibration_acyd"


@dataclass(frozen=True)
class Exp082Result:
    n_rows: int
    n_negatives: int
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    roc_auc_before: float
    roc_auc_after: float
    spearman_rho: float
    max_ece_after: float
    min_spearman_rho: float
    artifact_path: str
    passed: bool
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 with QML_DEVICE=cuda")


def gate_passed(result: Exp082Result) -> bool:
    return result.passed


def run_exp_082(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> Exp082Result:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    max_ece_after = float(cfg.get("max_ece_after", 0.08))
    min_spearman = float(cfg.get("min_spearman_rho", 0.85))
    min_auc_delta = float(cfg.get("min_auc_delta", -0.001))
    n_rows = int(cfg.get("n_rows", 2000))

    init_correlation_id()
    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP {EXP_ID} — Isotonic calibration (ACYD C4) | profile={profile}")
        print(f"Gates: ECE≤{max_ece_after} · ρ≥{min_spearman} · AUC Δ≥{min_auc_delta}")
        print(f"{'=' * 60}\n")

    outcome = run_calibration_evaluation(
        CalibrationEvaluationDTO(
            exp_id=str(cfg.get("model_exp_id", "exp_060")),
            model_name=str(cfg.get("model_name", "large_nano_mlp")),
            dataset=str(cfg.get("dataset_id", "acyd_soy_brazil_v1")),
            seed=int(cfg.get("seed", 42)),
            split=str(cfg.get("split", "val")),
            n_rows=n_rows,
            min_negatives=int(cfg.get("min_negatives", 50)),
            fit_fraction=float(cfg.get("fit_fraction", 0.8)),
            chunk_size=int(cfg.get("chunk_size", 2048)),
            force_balanced=bool(cfg.get("force_balanced", False)),
            ranking_domain="agro",
            artifact_exp_id=EXP_ID,
        ),
        min_spearman_rho=min_spearman,
        max_ece_after=max_ece_after,
        require_ece_improved=bool(cfg.get("require_ece_improved", True)),
        require_brier_improved=bool(cfg.get("require_brier_improved", True)),
        min_auc_delta=min_auc_delta,
    )
    if isinstance(outcome, Fail):
        raise RuntimeError(f"{outcome.error.code}: {outcome.error.message}")
    assert isinstance(outcome, Ok)
    result = outcome.value
    elapsed = time.perf_counter() - t0

    log = ExperimentLogger(EXP_ID, MODEL_NAME, seed=int(cfg.get("seed", 42)), profile=profile)
    log.log(
        1,
        ece_before=round(result.ece_before, 4),
        ece_after=round(result.ece_after, 4),
        brier_before=round(result.brier_before, 4),
        brier_after=round(result.brier_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
    )
    log.finish(
        elapsed,
        record_type="calibration_evaluation",
        n_rows=result.n_rows,
        n_negatives=result.n_negatives,
        ece_before=round(result.ece_before, 4),
        ece_after=round(result.ece_after, 4),
        brier_before=round(result.brier_before, 4),
        brier_after=round(result.brier_after, 4),
        roc_auc_before=round(result.roc_auc_before, 4),
        roc_auc_after=round(result.roc_auc_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
        artifact_path=result.artifact_path,
        passed=result.passed,
        eval_set="acyd_soy_val",
    )

    log_event(
        "info",
        "exp_082 calibration summary",
        exp_id=EXP_ID,
        profile=profile,
        ece_after=round(result.ece_after, 4),
        brier_after=round(result.brier_after, 4),
        spearman_rho=round(result.spearman_rho, 4),
        passed=result.passed,
    )

    out = Exp082Result(
        n_rows=result.n_rows,
        n_negatives=result.n_negatives,
        ece_before=result.ece_before,
        ece_after=result.ece_after,
        brier_before=result.brier_before,
        brier_after=result.brier_after,
        roc_auc_before=result.roc_auc_before,
        roc_auc_after=result.roc_auc_after,
        spearman_rho=result.spearman_rho,
        max_ece_after=max_ece_after,
        min_spearman_rho=min_spearman,
        artifact_path=result.artifact_path,
        passed=result.passed,
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if out.passed else "FAIL"
        print(
            f"n={out.n_rows} (neg={out.n_negatives}) | "
            f"ECE {out.ece_before:.4f}→{out.ece_after:.4f} | "
            f"Brier {out.brier_before:.4f}→{out.brier_after:.4f} | "
            f"AUC {out.roc_auc_before:.4f}→{out.roc_auc_after:.4f} | "
            f"ρ={out.spearman_rho:.4f} [{status}] | {out.elapsed_s}s",
            flush=True,
        )
        print(f"artifact → {out.artifact_path}", flush=True)

    return out


def _build_results_md(result: Exp082Result) -> str:
    verdict = "**accepted**" if result.passed else "**rejected**"
    return "\n".join(
        [
            "# Results — EXP 082: Isotonic Calibration (ACYD C4)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  ",
            "**Model:** `exp_060` LargeNanoMLP · `acyd_soy_brazil_v1` · seed 42",
            "",
            "## Summary",
            "",
            "| Metric | Value | Gate |",
            "|--------|-------|------|",
            f"| Val rows | **{result.n_rows}** | — |",
            f"| Negatives | **{result.n_negatives}** | — |",
            f"| ECE before | **{result.ece_before:.4f}** | — |",
            f"| ECE after | **{result.ece_after:.4f}** | ≤ {result.max_ece_after} · < before |",
            f"| Brier before | **{result.brier_before:.4f}** | — |",
            f"| Brier after | **{result.brier_after:.4f}** | ≤ before |",
            f"| ROC-AUC before | **{result.roc_auc_before:.4f}** | — |",
            f"| ROC-AUC after | **{result.roc_auc_after:.4f}** | Δ ≥ −0.005 |",
            f"| Spearman ρ (agro) | **{result.spearman_rho:.4f}** | ≥ {result.min_spearman_rho} |",
            f"| Elapsed | **{result.elapsed_s:.1f}s** | — |",
            "",
            "## Verdict",
            f"{verdict} — isotonic calibration on temporal ACYD val for Agro Risk Lab probabilities.",
            "",
            "## Artifact",
            "",
            f"- `{result.artifact_path}`",
            "",
            "## Limitations",
            "",
            "- Temporal val fit only; not operational ZARC / insurance advice.",
            "- Isotonic is monotone — ranking preservation is expected by construction.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 082 — isotonic calibration on ACYD C4")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_082(profile=args.profile, verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
