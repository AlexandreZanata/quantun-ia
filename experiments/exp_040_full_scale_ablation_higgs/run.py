"""
EXP 040 — Full-scale (805K) methodology ablation on HIGGS.

Reuses exp_036 trainers with profile=full_scale on the complete train split.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_040_full_scale_ablation_higgs/run.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import experiments.exp_036_method_ablation_higgs.run as ablation
from experiments.exp_036_method_ablation_higgs.run import METHODS, MethodAblationResult

EXP_KEY = "exp_040_full_scale_ablation_higgs"
EXP_ID = "exp_040"
PROFILE = "full_scale"


def run_exp_040(
    *,
    verbose: bool = True,
    require_cuda: bool = True,
) -> MethodAblationResult:
    prev_key = ablation.EXP_KEY
    prev_id = ablation.EXP_ID
    ablation.EXP_KEY = EXP_KEY
    ablation.EXP_ID = EXP_ID
    try:
        return ablation.run_exp_036(profile=PROFILE, verbose=verbose, require_cuda=require_cuda)
    finally:
        ablation.EXP_KEY = prev_key
        ablation.EXP_ID = prev_id


def _passed(result: MethodAblationResult) -> bool:
    return len(result.beaters) > 0


def _summarize(result: MethodAblationResult) -> str:
    verdict = "accepted" if _passed(result) else "rejected (honest negative)"
    lines = [
        f"\n{'=' * 60}",
        "EXP 040 SUMMARY",
        f"{'=' * 60}",
        f"Seeds: {result.n_seeds} | Train rows: {result.n_train_rows:,} | Val: {result.n_val_rows:,}",
    ]
    for method in METHODS:
        mean_auc = result.mean_auc_by_method[method]
        delta_pp = (mean_auc - result.baseline_mean_auc) * 100.0 if method != "baseline" else 0.0
        suffix = f" (Δ={delta_pp:+.2f} pp)" if method != "baseline" else ""
        lines.append(f"  {method}: {mean_auc:.4f}{suffix}")
    lines.extend(
        [
            f"Beaters (≥ {result.min_beat_baseline_pp} pp): {list(result.beaters) or 'none'}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def _build_results_md(result: MethodAblationResult) -> str:
    from datetime import date

    verdict = "accepted" if _passed(result) else "rejected (honest negative)"
    rows = [
        "# Results — EXP 040: Full-Scale HIGGS Methodology Ablation",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Mean val ROC-AUC by method (805K train)",
        "",
        "| Method | Mean val AUC | Δ vs baseline |",
        "|--------|--------------|---------------|",
    ]
    for method in METHODS:
        mean_auc = result.mean_auc_by_method[method]
        delta = (mean_auc - result.baseline_mean_auc) * 100.0
        delta_str = "—" if method == "baseline" else f"{delta:+.2f} pp"
        rows.append(f"| {method} | **{mean_auc:.4f}** | {delta_str} |")
    rows.extend(
        [
            "",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Seeds: **{result.n_seeds}**",
            f"- Beaters (≥ {result.min_beat_baseline_pp} pp): **{list(result.beaters) or 'none'}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — full-scale paired comparison vs Adam baseline on HIGGS.",
            "",
            "## Comparison to exp_036 (50K slice)",
            "- exp_036 best alternative: adaptive **+0.26 pp** (10 seeds, honest negative).",
            "",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 040 — Full-scale HIGGS methodology ablation")
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_040(verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
