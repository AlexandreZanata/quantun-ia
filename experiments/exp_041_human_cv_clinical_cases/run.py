"""
EXP 041 — Literature-backed clinical case validation (human-facing).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_041_human_cv_clinical_cases/run.py
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_041_human_cv_clinical_cases/run.py --write-results
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

from scipy.stats import spearmanr

from src.application.clinical_validation_cases import CLINICAL_VALIDATION_CASES, high_risk_cases, low_risk_cases
from src.application.human_cv_scorer import score_patient
from src.shared.result import Fail, Ok
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_ID = "exp_041"
MODEL_NAME = "large_nano_mlp_clinical_validation"
MIN_SPEARMAN = 0.85


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    title: str
    expected_tier: str
    expected_rank: int
    science_note: str
    risk_percent: float
    risk_band: str
    probability: float


@dataclass(frozen=True)
class ClinicalValidationResult:
    case_scores: tuple[CaseScore, ...]
    spearman_rho: float
    spearman_p: float
    max_low_risk_percent: float
    min_high_risk_percent: float
    separation_pp: float
    monotonic: bool
    passed: bool
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 with QML_DEVICE=cuda")


def _is_monotonic(scores: list[CaseScore], *, epsilon: float = 1e-4) -> bool:
    """Check non-decreasing risk by expected rank (epsilon tolerance at probability ceiling)."""
    ordered = sorted(scores, key=lambda c: c.expected_rank)
    probs = [c.probability for c in ordered]
    return all(probs[i] <= probs[i + 1] + epsilon for i in range(len(probs) - 1))


def run_exp_041(*, verbose: bool = True, require_cuda: bool = True) -> ClinicalValidationResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    init_correlation_id()
    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP {EXP_ID} — Clinical case validation (8 literature-backed profiles)")
        print(f"{'=' * 60}\n")

    scored: list[CaseScore] = []
    for case in CLINICAL_VALIDATION_CASES:
        outcome = score_patient(case.profile)
        if isinstance(outcome, Fail):
            raise RuntimeError(f"{case.case_id}: {outcome.error.code} — {outcome.error.message}")
        assert isinstance(outcome, Ok)
        r = outcome.value
        row = CaseScore(
            case_id=case.case_id,
            title=case.title,
            expected_tier=case.expected_tier,
            expected_rank=case.expected_rank,
            science_note=case.science_note,
            risk_percent=r.risk_percent,
            risk_band=r.risk_band,
            probability=r.probability,
        )
        scored.append(row)
        if verbose:
            print(
                f"  {case.case_id} rank={case.expected_rank} ({case.expected_tier:10s}) "
                f"→ {r.risk_percent:6.2f}% [{r.risk_band}]  {case.title}",
                flush=True,
            )

    ranks = [c.expected_rank for c in scored]
    risks = [c.risk_percent for c in scored]
    rho, p_value = spearmanr(ranks, risks)

    low = low_risk_cases()
    high = high_risk_cases()
    low_ids = {c.case_id for c in low}
    high_ids = {c.case_id for c in high}
    max_low_prob = max(c.probability for c in scored if c.case_id in low_ids)
    min_high_prob = min(c.probability for c in scored if c.case_id in high_ids)
    separation = (min_high_prob - max_low_prob) * 100.0
    monotonic = _is_monotonic(scored)

    passed = float(rho) >= MIN_SPEARMAN and separation > 0
    elapsed = time.perf_counter() - t0

    log = ExperimentLogger(EXP_ID, MODEL_NAME, seed=42, profile="clinical_validation")
    for idx, case in enumerate(scored, start=1):
        log.log(
            idx,
            case_id=case.case_id,
            expected_rank=case.expected_rank,
            expected_tier=case.expected_tier,
            risk_percent=round(case.risk_percent, 3),
            probability=round(case.probability, 4),
        )
    log.finish(
        elapsed,
        record_type="clinical_validation",
        spearman_rho=round(float(rho), 4),
        spearman_p=round(float(p_value), 6),
        max_low_risk_percent=round(max_low_prob * 100.0, 3),
        min_high_risk_percent=round(min_high_prob * 100.0, 3),
        separation_pp=round(separation, 3),
        monotonic=monotonic,
        n_cases=len(scored),
        passed=passed,
        eval_set="clinical_cases",
    )

    log_event(
        "info",
        "exp_041 clinical validation summary",
        exp_id=EXP_ID,
        spearman_rho=round(float(rho), 4),
        separation_pp=round(separation, 3),
        passed=passed,
    )

    if verbose:
        print(
            f"\nSpearman ρ={rho:.4f} (p={p_value:.4g}) | "
            f"separation={separation:+.2f} pp (min H − max L) | "
            f"monotonic={monotonic} | verdict={'PASS' if passed else 'FAIL'} | "
            f"elapsed={elapsed:.1f}s\n",
            flush=True,
        )

    return ClinicalValidationResult(
        case_scores=tuple(scored),
        spearman_rho=float(rho),
        spearman_p=float(p_value),
        max_low_risk_percent=max_low_prob * 100.0,
        min_high_risk_percent=min_high_prob * 100.0,
        separation_pp=separation,
        monotonic=monotonic,
        passed=passed,
        elapsed_s=elapsed,
    )


def _build_results_md(result: ClinicalValidationResult) -> str:
    verdict = "**accepted**" if result.passed else "**rejected**"
    lines = [
        "# Results — EXP 041: Human Clinical Case Validation",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  ",
        "**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42",
        "",
        "## Summary",
        "",
        f"| Metric | Value | Gate |",
        f"|--------|-------|------|",
        f"| Spearman ρ (rank vs risk %) | **{result.spearman_rho:.4f}** | ≥ {MIN_SPEARMAN} |",
        f"| Max risk — low tier (L01–L04) | **{result.max_low_risk_percent:.2f}%** | — |",
        f"| Min risk — high tier (H01–H04) | **{result.min_high_risk_percent:.2f}%** | — |",
        f"| Separation (min H − max L) | **{result.separation_pp:+.3f} pp** | > 0 |",
        f"| Strict monotonic (ε=1e-4) | **{result.monotonic}** | informational |",
        f"| Elapsed | **{result.elapsed_s:.1f}s** | — |",
        "",
        f"## Verdict",
        f"{verdict} — model ordering vs literature-backed clinical expectation.",
        "",
        "## Case-by-case results",
        "",
        "| ID | Expected rank | Tier | Model risk % | Band | Title |",
        "|----|---------------|------|--------------|------|-------|",
    ]
    for case in sorted(result.case_scores, key=lambda c: c.expected_rank):
        lines.append(
            f"| **{case.case_id}** | {case.expected_rank} | {case.expected_tier} | "
            f"**{case.risk_percent:.2f}%** | {case.risk_band} | {case.title} |"
        )
    lines.extend(
        [
            "",
            "## Science notes (expected direction)",
            "",
        ]
    )
    for case in sorted(result.case_scores, key=lambda c: c.expected_rank):
        lines.append(f"- **{case.case_id}** — {case.science_note}")
    lines.extend(
        [
            "",
            "## Interpretation for humans",
            "",
            "- **L01–L04** are profiles where a short-term CV event is *unlikely* by epidemiology.",
            "- **H01–H04** are profiles where an event is *likely* (prior ASCVD, multimorbidity, smoking).",
            "- The Synthea v1 cohort has ~99% positive prevalence, so absolute % looks high for everyone.",
            "- **What we validated:** the model *ranks* patients in clinically sensible order.",
            "- **Ceiling note:** H02–H04 saturate near 100%; tiny ordering noise there is not clinically meaningful.",
            "",
            "## Limitations",
            "",
            "- Synthetic training data; not calibrated for real-world prevalence.",
            "- Hand-entered profiles use `latent_risk_factor = 0` (neutral).",
            "- Does not replace ACC/AHA pooled cohort equations or clinician judgment.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 041 — clinical case validation")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_041(verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
