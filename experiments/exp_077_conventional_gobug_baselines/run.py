"""
EXP 077 — LargeNanoMLP vs conventional sklearn/XGBoost baselines on GoBug code defects.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_077_conventional_gobug_baselines/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.conventional_baselines import (
    ConventionalComparisonResult,
    gate_passed,
    run_conventional_gobug_comparison,
)
from src.training.metrics import ExperimentLogger
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_077_conventional_gobug_baselines"
EXP_ID = "exp_077"
ROOT = Path(__file__).resolve().parents[2]
PRIMARY_METRIC_LABEL = "PR-AUC"


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _log_scores(result: ConventionalComparisonResult, *, seed: int, profile: str) -> None:
    summary: dict[str, dict[str, float]] = {}
    for score in result.scores:
        logger = ExperimentLogger(
            EXP_ID,
            score.model_key,
            seed=seed,
            profile=profile,
        )
        logger.log(
            0,
            loss=0.0,
            accuracy=score.accuracy,
            pr_auc=score.roc_auc,
        )
        logger.finish(
            score.train_s,
            test_accuracy=score.accuracy,
            pr_auc=score.roc_auc,
            n_params=score.n_params,
            train_s=score.train_s,
            source=score.source,
            eval_set="holdout_val",
        )
        summary[score.model_key] = {
            "mean": score.roc_auc,
            "std": 0.0,
            "ci_low": score.roc_auc,
            "ci_high": score.roc_auc,
            "n_seeds": 1,
            "accuracy": score.accuracy,
        }

    record = {
        "exp_id": EXP_ID,
        "model_name": f"{EXP_ID}_pr_auc_summary",
        "record_type": "multi_seed_summary",
        "profile": profile,
        "seed": seed,
        "summary": summary,
        "best_conventional_pr_auc": result.best_conventional_auc,
        "nano_pr_auc": result.nano_auc,
        "advantage_pp": result.advantage_vs_best_conventional_pp,
    }
    logs_path = ROOT / "logs" / "experiments.jsonl"
    logs_path.parent.mkdir(exist_ok=True)
    with open(logs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_exp_077(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ConventionalComparisonResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 077 — Conventional GoBug baselines | profile={profile}")
        print(f"Gate: LargeNanoMLP ≥ best conventional + 0.5 pp {PRIMARY_METRIC_LABEL}")
        print(f"{'=' * 60}\n")

    result = run_conventional_gobug_comparison(ROOT, profile=profile)
    _log_scores(result, seed=42, profile=profile)

    log_event(
        "info",
        "exp_077 conventional baseline summary",
        exp_id=EXP_ID,
        profile=profile,
        nano_pr_auc=round(result.nano_auc, 4),
        best_conventional_pr_auc=round(result.best_conventional_auc, 4),
        advantage_pp=round(result.advantage_vs_best_conventional_pp, 3),
        n_train_rows=result.n_train_rows,
        n_val_rows=result.n_val_rows,
        elapsed_s=result.elapsed_s,
    )

    if verbose:
        best = max(result.scores, key=lambda s: s.roc_auc)
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"{'Model':<42} {PRIMARY_METRIC_LABEL:>8} {'Acc':>8} {'Train(s)':>10}",
            flush=True,
        )
        print("-" * 72, flush=True)
        for score in sorted(result.scores, key=lambda s: -s.roc_auc):
            print(
                f"{score.display_name:<42} {score.roc_auc:>8.4f} "
                f"{score.accuracy:>8.4f} {score.train_s:>10.1f}",
                flush=True,
            )
        print(
            f"\nBest conventional: {result.best_conventional_auc:.4f} | "
            f"LargeNanoMLP: {result.nano_auc:.4f} | "
            f"Δ={result.advantage_vs_best_conventional_pp:+.2f} pp [{status}] | "
            f"elapsed={result.elapsed_s}s",
            flush=True,
        )
        if best.model_key != "large_nano_mlp":
            print(f"Warning: best overall model is {best.display_name}", flush=True)

    return result


def _build_results_md(result: ConventionalComparisonResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected"
    lines = [
        "# Results — EXP 077: Conventional tabular baselines vs LargeNanoMLP (GoBug code defects)",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Validation gate (PR-AUC primary)",
        "",
        f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
        f"- Best conventional val PR-AUC: **{result.best_conventional_auc:.4f}**",
        f"- LargeNanoMLP val PR-AUC: **{result.nano_auc:.4f}**",
        f"- Advantage: **{result.advantage_vs_best_conventional_pp:+.2f} pp** "
        f"(gate ≥ **{result.min_advantage_pp}** pp)",
        f"- Elapsed: **{result.elapsed_s}s**",
        "",
        "| Model | Val PR-AUC | Val accuracy | Train (s) |",
        "|-------|------------|--------------|-----------|",
    ]
    for score in sorted(result.scores, key=lambda s: -s.roc_auc):
        lines.append(
            f"| {score.display_name} | {score.roc_auc:.4f} | "
            f"{score.accuracy:.4f} | {score.train_s:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            f"**{verdict}** — LargeNanoMLP vs conventional sklearn/XGBoost on GoBug "
            f"(`profile={result.profile}`).",
            "",
            "## Limitations",
            "- Single seed; temporal val split only (aligned with exp_070).",
            "- LargeNanoMLP evaluated from exp_070 checkpoint; baselines retrained each run.",
            "- Software defect benchmark — not production static analysis.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 077 — conventional GoBug baselines")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_077(profile=args.profile, verbose=not args.quiet)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
