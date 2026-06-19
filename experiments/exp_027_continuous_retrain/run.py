"""
EXP 027 — Continuous retrain champion/challenger gate (4 simulated weekly cycles).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_027_continuous_retrain/run.py --profile ci
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.continuous_train import run_continuous_train
from src.training.champion import champion_dir, load_champion_manifest
from src.training.config import load_experiment_config

EXP_KEY = "exp_027_continuous_retrain"
EXP_ID = "exp_027"
MODEL = "hybrid_sandwich"
DATASET = "breast_cancer"
CHAMPION_EXP_ID = "quantum_nano_bc_app"


@dataclass(frozen=True)
class WeekResult:
    week: int
    seed: int
    champion_accuracy: float
    challenger_accuracy: float
    delta_pp: float
    promoted: bool
    blocked: bool
    elapsed_s: float


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _bootstrap_champion(
    *,
    profile: str,
    epochs: int,
    champion_seed: int,
    verbose: bool,
) -> None:
    """Train and promote a fresh champion before weekly challenger cycles."""
    root = champion_dir()
    if root.exists():
        shutil.rmtree(root)
    if verbose:
        print(f"Bootstrapping champion (seed={champion_seed})...", flush=True)
    run_continuous_train(
        profile=profile,
        epochs=epochs,
        seed=champion_seed,
        challenger_exp_id=CHAMPION_EXP_ID,
        champion_exp_id=CHAMPION_EXP_ID,
        champion_seed=champion_seed,
        bootstrap_only=True,
    )


def run_exp_027(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
    fresh_champion: bool = True,
) -> list[WeekResult]:
    """Run simulated weekly retrain cycles with champion/challenger comparison."""
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds: list[int] = list(cfg["seeds"])
    epochs: int = int(cfg["epochs"])
    weeks: int = int(cfg.get("weeks", 4))
    promote_max = float(cfg.get("promote_max_delta_pp", 0.5))
    rollback_max = float(cfg.get("demote_regression_pp", 1.0))
    champion_seed = int(cfg.get("champion_seed", 42))

    cycle_seeds = [s for s in seeds if s != champion_seed][:weeks]
    if len(cycle_seeds) < weeks:
        raise ValueError(f"need at least {weeks} challenger seeds (≠ champion), got {len(cycle_seeds)}")

    if fresh_champion:
        _bootstrap_champion(
            profile=profile,
            epochs=epochs,
            champion_seed=champion_seed,
            verbose=verbose,
        )

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 027 — Continuous Retrain | profile={profile} | weeks={weeks}")
        print(f"Model: {MODEL} × {DATASET} | epochs={epochs} | promote≤{promote_max} pp")
        print(f"{'=' * 60}\n")

    results: list[WeekResult] = []
    for week, seed in enumerate(cycle_seeds, start=1):
        if verbose:
            print(f"[Week {week}/{weeks}] seed={seed} — training challenger...", flush=True)
        t0 = time.perf_counter()
        outcome = run_continuous_train(
            model_name=MODEL,
            dataset=DATASET,
            profile=profile,
            epochs=epochs,
            seed=seed,
            challenger_exp_id=EXP_ID,
            champion_exp_id=CHAMPION_EXP_ID,
            champion_seed=None,
            promote_max_delta_pp=promote_max,
            rollback_regression_pp=rollback_max,
        )
        elapsed = time.perf_counter() - t0
        week_result = WeekResult(
            week=week,
            seed=seed,
            champion_accuracy=outcome.champion_accuracy,
            challenger_accuracy=outcome.challenger_accuracy,
            delta_pp=outcome.delta_pp,
            promoted=outcome.promoted,
            blocked=outcome.blocked,
            elapsed_s=round(elapsed, 2),
        )
        results.append(week_result)
        if verbose:
            status = "PROMOTED" if outcome.promoted else ("BLOCKED" if outcome.blocked else "KEPT")
            print(
                f"         {status} — champion={outcome.champion_accuracy * 100:.2f}% "
                f"challenger={outcome.challenger_accuracy * 100:.2f}% "
                f"Δ={outcome.delta_pp:.2f} pp ({elapsed:.1f}s)",
                flush=True,
            )

    manifest = load_champion_manifest()
    if manifest is None:
        raise RuntimeError("champion manifest missing after exp_027")
    link = champion_dir() / "checkpoint"
    if not link.is_symlink() and not link.is_dir():
        raise RuntimeError("champion checkpoint symlink missing")

    return results


def _summarize(results: list[WeekResult]) -> str:
    promoted = sum(1 for r in results if r.promoted)
    blocked = sum(1 for r in results if r.blocked)
    within = sum(1 for r in results if r.delta_pp <= 0.5)
    non_blocked = len(results) - blocked
    accepted = blocked <= 1 and non_blocked >= min(3, len(results))
    verdict = "accepted" if accepted else "rejected"

    lines = [
        f"\n{'=' * 60}",
        "EXP 027 SUMMARY",
        f"{'=' * 60}",
        f"{'Week':>5}  {'Seed':>6}  {'Champ %':>8}  {'Chall %':>8}  {'Δ pp':>7}  {'Decision':>10}",
        "-" * 60,
    ]
    for r in results:
        decision = "PROMOTED" if r.promoted else ("BLOCKED" if r.blocked else "KEPT")
        lines.append(
            f"{r.week:>5}  {r.seed:>6}  {r.champion_accuracy * 100:>7.2f}%  "
            f"{r.challenger_accuracy * 100:>7.2f}%  {r.delta_pp:>6.2f}  {decision:>10}"
        )
    lines.extend(
        [
            "-" * 60,
            f"Promoted: {promoted} | Blocked: {blocked} | Within 0.5 pp: {within}/{len(results)}",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 027 — continuous retrain gate")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    results = run_exp_027(profile=args.profile, verbose=not args.quiet)
    summary = _summarize(results)
    print(summary)

    if args.write_results:
        out_dir = Path(__file__).resolve().parent
        content = _build_results_md(results)
        (out_dir / "results.md").write_text(content, encoding="utf-8")
        print(f"Wrote {out_dir / 'results.md'}")

    blocked = sum(1 for r in results if r.blocked)
    non_blocked = len(results) - blocked
    ok = blocked <= 1 and non_blocked >= min(3, len(results))
    return 0 if ok else 1


def _build_results_md(results: list[WeekResult]) -> str:
    from datetime import date

    lines = [
        "# Results — EXP 027: Continuous Retrain Champion/Challenger Gate",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Weekly cycles",
        "",
        "| Week | Seed | Champion % | Challenger % | Δ pp | Decision |",
        "|------|------|------------|--------------|------|----------|",
    ]
    for r in results:
        decision = "PROMOTED" if r.promoted else ("BLOCKED" if r.blocked else "KEPT")
        lines.append(
            f"| {r.week} | {r.seed} | {r.champion_accuracy * 100:.2f} | "
            f"{r.challenger_accuracy * 100:.2f} | {r.delta_pp:.2f} | {decision} |"
        )

    blocked = sum(1 for r in results if r.blocked)
    within = sum(1 for r in results if r.delta_pp <= 0.5)
    non_blocked = len(results) - blocked
    verdict = "accepted" if blocked <= 1 and non_blocked >= min(3, len(results)) else "rejected"
    lines.extend(
        [
            "",
            "## Verdict",
            f"**{verdict}** — blocked cycles: {blocked}; within 0.5 pp: {within}/{len(results)}; "
            f"non-blocked: {non_blocked}/{len(results)}.",
            "",
            "## Conclusion",
            "Champion/challenger gate with `artifacts/champion/` symlink operates as designed "
            "for simulated weekly retrain on breast cancer holdout.",
            "",
            "## Limitations",
            "- Simulated weeks (sequential runs), not wall-clock cron.",
            "- Single dataset; not a clinical deployment claim.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
