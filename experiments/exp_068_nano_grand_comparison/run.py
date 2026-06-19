"""
EXP 068 — Grand comparison synthesis across C1–C4 classical vs quantum recipes.

Aggregates curated publication metrics (no GPU training):
  MLFLOW_DISABLE=1 python experiments/exp_068_nano_grand_comparison/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.config import load_experiment_config
from src.training.grand_comparison import (
    build_grand_comparison_matrix,
    export_grand_comparison_json,
    export_grand_comparison_latex,
    grand_comparison_to_dict,
    load_grand_comparison_registry,
)
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_068_nano_grand_comparison"
EXP_ID = "exp_068"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GrandComparisonRunResult:
    json_path: Path
    latex_path: Path
    n_recipes: int
    n_domains: int
    hypothesis_confirmed: bool
    quantum_recipe_wins: dict[str, int]
    elapsed_s: float


def _append_jsonl_summary(
    result_cells: tuple,
    *,
    profile: str,
    seed: int,
    elapsed_s: float,
    hypothesis_confirmed: bool,
) -> None:
    from datetime import datetime

    record = {
        "exp_id": EXP_ID,
        "model_name": f"{EXP_ID}_grand_comparison",
        "started_at": datetime.now().isoformat(),
        "profile": profile,
        "seed": seed,
        "elapsed_s": elapsed_s,
        "hypothesis_confirmed": hypothesis_confirmed,
        "eval_set": "synthesis",
        "artifact_json": "dist/leaderboards/nano_grand_comparison.json",
        "artifact_latex": "paper/tables/grand_comparison.tex",
        "n_cells": len([c for c in result_cells if c.delta_pp is not None]),
    }
    logs_path = ROOT / "logs" / "experiments.jsonl"
    logs_path.parent.mkdir(exist_ok=True)
    with open(logs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_exp_068(
    *,
    profile: str = "ci",
    verbose: bool = True,
) -> GrandComparisonRunResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    claim_win_delta_pp = float(cfg.get("claim_win_delta_pp", 0.5))
    registry_path = ROOT / str(cfg.get("registry_path", "config/grand_comparison_registry.yaml"))
    json_out = ROOT / str(cfg.get("json_out", "dist/leaderboards/nano_grand_comparison.json"))
    latex_out = ROOT / str(cfg.get("latex_out", "paper/tables/grand_comparison.tex"))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 068 — Nano grand comparison synthesis | profile={profile}")
        print(f"Registry: {registry_path.name}")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    registry = load_grand_comparison_registry(registry_path)
    matrix_result = build_grand_comparison_matrix(registry, claim_win_delta_pp=claim_win_delta_pp)
    payload = grand_comparison_to_dict(matrix_result, registry)

    export_grand_comparison_json(payload, json_out)
    export_grand_comparison_latex(matrix_result, registry, latex_out)

    elapsed = time.perf_counter() - t0
    _append_jsonl_summary(
        matrix_result.cells,
        profile=profile,
        seed=seed,
        elapsed_s=elapsed,
        hypothesis_confirmed=matrix_result.hypothesis_confirmed,
    )

    log_event(
        "info",
        "exp_068 grand comparison complete",
        exp_id=EXP_ID,
        profile=profile,
        json_path=str(json_out),
        latex_path=str(latex_out),
        hypothesis_confirmed=matrix_result.hypothesis_confirmed,
        quantum_recipe_wins=matrix_result.quantum_recipe_wins,
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        print(f"Wrote {json_out}")
        print(f"Wrote {latex_out}")
        print(f"Hypothesis confirmed: {matrix_result.hypothesis_confirmed}")
        for recipe, wins in matrix_result.quantum_recipe_wins.items():
            if registry["recipes"][recipe].get("quantum"):
                print(f"  {recipe}: {wins}/4 domain wins (≥ {claim_win_delta_pp} pp)")
        if matrix_result.pending_domains:
            print(f"Pending: {matrix_result.pending_domains}")
        print(f"Elapsed: {elapsed:.2f}s\n")

    return GrandComparisonRunResult(
        json_path=json_out,
        latex_path=latex_out,
        n_recipes=len(matrix_result.recipes),
        n_domains=len(matrix_result.domains),
        hypothesis_confirmed=matrix_result.hypothesis_confirmed,
        quantum_recipe_wins=matrix_result.quantum_recipe_wins,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: GrandComparisonRunResult) -> bool:
    return result.json_path.is_file() and result.latex_path.is_file()


def _build_results_md(result: GrandComparisonRunResult) -> str:
    wins_lines = [
        f"- `{recipe}`: **{wins}** domain wins"
        for recipe, wins in sorted(result.quantum_recipe_wins.items())
    ]
    verdict = "confirmed" if result.hypothesis_confirmed else "rejected"
    return "\n".join(
        [
            "# Results — EXP 068: Nano Grand Comparison (C1–C4 Synthesis)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs)",
            "",
            "## Artifacts",
            "",
            f"- JSON leaderboard: `{result.json_path.relative_to(ROOT)}`",
            f"- LaTeX table: `{result.latex_path.relative_to(ROOT)}`",
            f"- Recipes: **{result.n_recipes}** · Domains: **{result.n_domains}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Quantum recipe wins (≥ +0.5 pp)",
            "",
            *wins_lines,
            "",
            "## Verdict",
            f"**Hypothesis {verdict}** — no quantum recipe wins on ≥3/4 domains simultaneously.",
            "",
            "## Limitations",
            "- Curated single-seed publication metrics; GoBug QNN head pending exp_071.",
            "- Synthesis only — does not re-train models.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 068 — Nano grand comparison synthesis")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_068(profile=args.profile, verbose=not args.quiet)

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
