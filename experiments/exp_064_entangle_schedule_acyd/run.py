"""
EXP 064 — Dynamic entanglement schedule vs fixed topologies on ACYD (C4 / H-Q3).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_064_entangle_schedule_acyd/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_parquet import load_open_parquet_splits
from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.config import load_experiment_config
from src.training.entangle_schedule import (
    DEFAULT_ENTANGLEMENT_LADDER,
    train_entangled_schedule,
    train_fixed_entangled,
)
from src.training.holdout import compare_conditions_batch
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_064_entangle_schedule_acyd"
EXP_ID = "exp_064"
FIXED_TOPOLOGIES = ("none", "chain", "ring")
ROOT = Path(__file__).resolve().parents[2]

_METRIC_LABELS = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "accuracy": "accuracy"}


@dataclass(frozen=True)
class EntangleScheduleAcydResult:
    n_seeds: int
    n_train_rows: int
    n_val_rows: int
    n_stages: int
    epochs_per_stage: int
    holdout_metric: str
    mean_schedule: float
    mean_by_topology: dict[str, float]
    best_fixed_topology: str
    best_fixed_mean: float
    advantage_pp: float
    min_advantage_pp: float
    paired_wins: int
    wilcoxon_p: float | None
    schedule_scores: tuple[float, ...]
    best_fixed_scores: tuple[float, ...]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _build_model(input_dim: int, cfg: dict, entanglement: str) -> QuantumNetEntangled:
    return QuantumNetEntangled(
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        entanglement=entanglement,
        input_dim=input_dim,
        reupload=bool(cfg.get("reupload", True)),
    )


def gate_passed(result: EntangleScheduleAcydResult) -> bool:
    return result.advantage_pp >= result.min_advantage_pp


def run_exp_064(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> EntangleScheduleAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seeds = tuple(int(s) for s in cfg.get("seeds", [42]))
    n_stages = int(cfg.get("n_stages", 5))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 10))
    min_advantage_pp = float(cfg.get("min_advantage_pp", 0.5))
    lr = float(cfg.get("learning_rate", 0.02))
    holdout_metric = str(cfg.get("holdout_metric", "roc_auc"))
    ladder = tuple(cfg.get("entanglement_ladder", DEFAULT_ENTANGLEMENT_LADDER))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 064 — Entanglement schedule vs fixed (ACYD) | profile={profile} | "
            f"seeds={len(seeds)} | stages={n_stages} × {epochs_per_stage}ep"
        )
        print(f"Metric: {holdout_metric} | Gate: schedule ≥ best fixed + {min_advantage_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    schedule_scores: list[float] = []
    fixed_by_topo: dict[str, list[float]] = {t: [] for t in FIXED_TOPOLOGIES}

    x_train, y_train, x_val, y_val, _x_test, _y_test, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seeds[0],
    )
    input_dim = int(x_train.shape[1])
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    for seed in seeds:
        def build(ent: str) -> QuantumNetEntangled:
            return _build_model(input_dim, cfg, ent)

        if verbose:
            print(f"Seed {seed}: schedule...", flush=True)
        sched_score = train_entangled_schedule(
            build,
            x_train_t,
            y_train_t,
            x_val_t,
            y_val_t,
            EXP_ID,
            f"entangle_schedule_seed{seed}",
            n_stages=n_stages,
            epochs_per_stage=epochs_per_stage,
            ladder=ladder,
            lr=lr,
            seed=seed,
            profile=profile,
            metric=holdout_metric,
        )
        schedule_scores.append(sched_score)

        for topo in FIXED_TOPOLOGIES:
            if verbose:
                print(f"Seed {seed}: fixed {topo}...", flush=True)
            score = train_fixed_entangled(
                build,
                topo,
                x_train_t,
                y_train_t,
                x_val_t,
                y_val_t,
                EXP_ID,
                f"entangle_{topo}_seed{seed}",
                n_stages=n_stages,
                epochs_per_stage=epochs_per_stage,
                lr=lr,
                seed=seed,
                profile=profile,
                metric=holdout_metric,
            )
            fixed_by_topo[topo].append(score)

        if verbose:
            best_seed = max(FIXED_TOPOLOGIES, key=lambda t: fixed_by_topo[t][-1])
            print(
                f"Seed {seed}: schedule={sched_score:.4f} | "
                f"best fixed ({best_seed})={fixed_by_topo[best_seed][-1]:.4f}",
                flush=True,
            )

    mean_by_topo = {t: statistics.mean(fixed_by_topo[t]) for t in FIXED_TOPOLOGIES}
    best_topo = max(mean_by_topo, key=mean_by_topo.get)
    best_fixed_mean = mean_by_topo[best_topo]
    mean_schedule = statistics.mean(schedule_scores)
    advantage_pp = (mean_schedule - best_fixed_mean) * 100.0
    best_fixed_scores = fixed_by_topo[best_topo]
    paired_wins = sum(1 for s, f in zip(schedule_scores, best_fixed_scores) if s > f)
    elapsed = time.perf_counter() - t0

    comparison = compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "entangle_schedule",
                "label_b": f"entangle_{best_topo}",
                "condition_a": schedule_scores,
                "condition_b": best_fixed_scores,
            }
        ],
    )
    wilcoxon_p = comparison[0].get("p_value") if comparison else None

    log_event(
        "info",
        "exp_064 entangle schedule summary",
        exp_id=EXP_ID,
        profile=profile,
        n_seeds=len(seeds),
        mean_schedule=round(mean_schedule, 4),
        best_fixed_topology=best_topo,
        best_fixed_mean=round(best_fixed_mean, 4),
        advantage_pp=round(advantage_pp, 3),
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
        elapsed_s=round(elapsed, 3),
    )

    result = EntangleScheduleAcydResult(
        n_seeds=len(seeds),
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        holdout_metric=holdout_metric,
        mean_schedule=mean_schedule,
        mean_by_topology=mean_by_topo,
        best_fixed_topology=best_topo,
        best_fixed_mean=best_fixed_mean,
        advantage_pp=advantage_pp,
        min_advantage_pp=min_advantage_pp,
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
        schedule_scores=tuple(schedule_scores),
        best_fixed_scores=tuple(best_fixed_scores),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"\nMean schedule={mean_schedule:.4f} | best fixed ({best_topo})={best_fixed_mean:.4f} | "
            f"Δ={advantage_pp:.2f} pp | wins={paired_wins}/{len(seeds)} [{status}] | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return result


def _summarize(result: EntangleScheduleAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    topo_lines = "\n".join(
        f"  {t}: {result.mean_by_topology[t]:.4f}"
        for t in FIXED_TOPOLOGIES
    )
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 064 SUMMARY",
            f"{'=' * 60}",
            f"Seeds: {result.n_seeds}",
            f"Stages: {result.n_stages} × {result.epochs_per_stage} epochs",
            f"Metric: {result.holdout_metric}",
            f"Mean schedule: {result.mean_schedule:.4f}",
            "Fixed topologies:",
            topo_lines,
            f"Best fixed: {result.best_fixed_topology} ({result.best_fixed_mean:.4f})",
            f"Advantage: {result.advantage_pp:+.2f} pp (gate ≥ {result.min_advantage_pp} pp)",
            f"Paired wins: {result.paired_wins}/{result.n_seeds}",
            f"Wilcoxon p: {result.wilcoxon_p}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: EntangleScheduleAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    metric_label = _METRIC_LABELS.get(result.holdout_metric, result.holdout_metric)
    return "\n".join(
        [
            "# Results — EXP 064: Dynamic entanglement schedule on ACYD (C4 / H-Q3)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Dataset:** `acyd_soy_brazil_v1` (val {metric_label})",
            "",
            f"## Validation gate ({metric_label})",
            "",
            f"- Seeds: **{result.n_seeds}**",
            f"- Train rows: **{result.n_train_rows:,}** · Val rows: **{result.n_val_rows:,}**",
            f"- Schedule stages: **{result.n_stages}** × **{result.epochs_per_stage}** epochs",
            f"- Mean schedule val: **{result.mean_schedule:.4f}**",
            f"- Best fixed ({result.best_fixed_topology}): **{result.best_fixed_mean:.4f}**",
            f"- Advantage: **{result.advantage_pp:+.2f} pp**",
            f"- Paired wins: **{result.paired_wins}/{result.n_seeds}**",
            f"- Wilcoxon p: **{result.wilcoxon_p}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — dynamic entanglement vs best fixed topology "
            f"(gate ≥ {result.min_advantage_pp} pp).",
            "",
            "## Limitations",
            "- Standalone `QuantumNetEntangled` (mirrors exp_053/exp_074); not frozen C4 backbone.",
            "- PennyLane QNN sim on CPU; publication row cap for feasible epoch cost.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 064 — dynamic entanglement schedule on ACYD (C4 / H-Q3)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_064(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
