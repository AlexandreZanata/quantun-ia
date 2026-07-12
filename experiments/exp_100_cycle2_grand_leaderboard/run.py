"""
EXP 100 — Cycle v2 grand leaderboard synthesis (exp_084–099).

Aggregates curated RTX 4060 publication metrics (no new training):
  MLFLOW_DISABLE=1 python experiments/exp_100_cycle2_grand_leaderboard/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.config import load_experiment_config
from src.training.cycle2_grand_leaderboard import (
    build_cycle2_grand_leaderboard,
    cycle2_leaderboard_to_dict,
    export_cycle2_grand_leaderboard_json,
    export_cycle2_grand_leaderboard_latex,
    load_cycle2_grand_leaderboard_registry,
)
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_100_cycle2_grand_leaderboard"
EXP_ID = "exp_100"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Cycle2GrandLeaderboardRunResult:
    json_path: Path
    latex_path: Path
    n_rows: int
    n_accepted: int
    n_rejected: int
    hypothesis_confirmed: bool
    quantum_claim_wins: tuple[str, ...]
    observed_accepts: frozenset[str]
    elapsed_s: float
    profile: str


def _append_jsonl_summary(result: Cycle2GrandLeaderboardRunResult, *, seed: int) -> None:
    record = {
        "exp_id": EXP_ID,
        "model_name": f"{EXP_ID}_cycle2_grand_leaderboard",
        "started_at": datetime.now().isoformat(),
        "profile": result.profile,
        "seed": seed,
        "elapsed_s": result.elapsed_s,
        "hypothesis_confirmed": result.hypothesis_confirmed,
        "eval_set": "synthesis",
        "n_rows": result.n_rows,
        "n_accepted": result.n_accepted,
        "n_rejected": result.n_rejected,
        "observed_accepts": sorted(result.observed_accepts),
        "quantum_claim_wins": list(result.quantum_claim_wins),
        "artifact_json": "dist/leaderboards/cycle2_grand_leaderboard.json",
        "artifact_latex": "paper/tables/cycle2_grand_leaderboard.tex",
    }
    logs_path = ROOT / "logs" / "experiments.jsonl"
    logs_path.parent.mkdir(exist_ok=True)
    with open(logs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_exp_100(
    *,
    profile: str = "ci",
    verbose: bool = True,
) -> Cycle2GrandLeaderboardRunResult:
    os.environ.setdefault("MLFLOW_DISABLE", "1")
    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    claim_win_delta_pp = float(cfg.get("claim_win_delta_pp", 0.5))
    registry_path = ROOT / str(
        cfg.get("registry_path", "config/cycle2_grand_leaderboard_registry.yaml")
    )
    json_out = ROOT / str(cfg.get("json_out", "dist/leaderboards/cycle2_grand_leaderboard.json"))
    latex_out = ROOT / str(cfg.get("latex_out", "paper/tables/cycle2_grand_leaderboard.tex"))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 100 — Cycle v2 grand leaderboard | profile={profile}")
        print(f"Registry: {registry_path.name}")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    registry = load_cycle2_grand_leaderboard_registry(registry_path)
    matrix = build_cycle2_grand_leaderboard(registry, claim_win_delta_pp=claim_win_delta_pp)
    payload = cycle2_leaderboard_to_dict(matrix, registry)

    export_cycle2_grand_leaderboard_json(payload, json_out)
    export_cycle2_grand_leaderboard_latex(matrix, latex_out)

    elapsed = time.perf_counter() - t0
    result = Cycle2GrandLeaderboardRunResult(
        json_path=json_out,
        latex_path=latex_out,
        n_rows=len(matrix.rows),
        n_accepted=sum(1 for r in matrix.rows if r.verdict == "accepted"),
        n_rejected=sum(1 for r in matrix.rows if r.verdict == "rejected"),
        hypothesis_confirmed=matrix.hypothesis_confirmed,
        quantum_claim_wins=matrix.quantum_claim_wins,
        observed_accepts=matrix.observed_accepts,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )
    _append_jsonl_summary(result, seed=seed)

    log_event(
        "info",
        "exp_100 cycle2 grand leaderboard complete",
        exp_id=EXP_ID,
        profile=profile,
        json_path=str(json_out),
        latex_path=str(latex_out),
        hypothesis_confirmed=result.hypothesis_confirmed,
        n_accepted=result.n_accepted,
        quantum_claim_wins=list(result.quantum_claim_wins),
        elapsed_s=result.elapsed_s,
    )

    if verbose:
        print(f"Wrote {json_out}")
        print(f"Wrote {latex_out}")
        print(f"Rows: {result.n_rows} · accepted: {result.n_accepted} · rejected: {result.n_rejected}")
        print(f"Accepts: {sorted(result.observed_accepts)}")
        print(f"Quantum claim wins (≥ +{claim_win_delta_pp} pp): {list(result.quantum_claim_wins) or 'none'}")
        print(f"Hypothesis confirmed: {result.hypothesis_confirmed}")
        print(f"Elapsed: {result.elapsed_s}s\n")

    return result


def gate_passed(result: Cycle2GrandLeaderboardRunResult) -> bool:
    return (
        result.json_path.is_file()
        and result.latex_path.is_file()
        and result.hypothesis_confirmed
    )


def _build_results_md(result: Cycle2GrandLeaderboardRunResult) -> str:
    accepts = ", ".join(sorted(result.observed_accepts)) or "(none)"
    verdict = "confirmed" if result.hypothesis_confirmed else "rejected"
    return "\n".join(
        [
            "# Results — EXP 100: Cycle v2 Grand Leaderboard",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** CPU synthesis (aggregates RTX 4060 publication runs)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Artifacts",
            "",
            f"- JSON leaderboard: `{result.json_path.relative_to(ROOT)}`",
            f"- LaTeX table: `{result.latex_path.relative_to(ROOT)}`",
            f"- Rows: **{result.n_rows}** · accepted: **{result.n_accepted}** · rejected: **{result.n_rejected}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Science gates",
            "",
            f"- Accepts: **{accepts}**",
            f"- Quantum claim wins (≥ +0.5 pp): **{list(result.quantum_claim_wins) or 'none'}**",
            "",
            "## Verdict",
            f"**Hypothesis {verdict}** — Cycle v2 scorecard closed; no quantum +0.5 pp claim win.",
            "",
            "## Limitations",
            "- Curated single-seed metrics from closed `results.md` files.",
            "- Synthesis only — does not re-train models.",
            "- Zenodo DOI / arXiv ID paste remains external.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 100 — Cycle v2 grand leaderboard synthesis")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_100(profile=args.profile, verbose=not args.quiet)

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
