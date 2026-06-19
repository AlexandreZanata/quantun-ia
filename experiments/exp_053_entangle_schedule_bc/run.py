"""
EXP 053 — Dynamic entanglement schedule vs fixed topologies on breast cancer.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_053_entangle_schedule_bc/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_entangled import QuantumNetEntangled
from src.training.config import load_experiment_config
from src.training.entangle_schedule import (
    DEFAULT_ENTANGLEMENT_LADDER,
    train_entangled_schedule,
    train_fixed_entangled,
)
from src.training.holdout import compare_conditions_batch
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_053_entangle_schedule_bc"
EXP_ID = "exp_053"
FIXED_TOPOLOGIES = ("none", "chain", "ring")


@dataclass(frozen=True)
class EntangleScheduleResult:
    n_seeds: int
    n_train_rows: int
    n_holdout_rows: int
    n_stages: int
    epochs_per_stage: int
    mean_schedule: float
    mean_by_topology: dict[str, float]
    best_fixed_topology: str
    best_fixed_mean: float
    advantage_pp: float
    min_advantage_pp: float
    paired_wins: int
    wilcoxon_p: float | None
    schedule_accuracies: tuple[float, ...]
    best_fixed_accuracies: tuple[float, ...]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _build_model(input_dim: int, cfg: dict, entanglement: str) -> QuantumNetEntangled:
    return QuantumNetEntangled(
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        entanglement=entanglement,
        input_dim=input_dim,
        reupload=bool(cfg.get("reupload", True)),
    )


def gate_passed(result: EntangleScheduleResult) -> bool:
    return result.advantage_pp >= result.min_advantage_pp


def run_exp_053(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> EntangleScheduleResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds = tuple(int(s) for s in cfg.get("seeds", [42]))
    n_stages = int(cfg.get("n_stages", 5))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 10))
    min_advantage_pp = float(cfg.get("min_advantage_pp", 1.0))
    lr = float(cfg.get("learning_rate", 0.02))
    ladder = tuple(cfg.get("entanglement_ladder", DEFAULT_ENTANGLEMENT_LADDER))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 053 — Entanglement schedule vs fixed | profile={profile} | "
            f"seeds={len(seeds)} | stages={n_stages} × {epochs_per_stage}ep"
        )
        print(f"Gate: schedule ≥ best fixed + {min_advantage_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    schedule_accs: list[float] = []
    fixed_by_topo: dict[str, list[float]] = {t: [] for t in FIXED_TOPOLOGIES}

    for seed in seeds:
        x_train, x_holdout, y_train, y_holdout, meta = prepare_dataset(
            str(cfg.get("dataset", "breast_cancer")),
            random_state=seed,
        )
        input_dim = int(meta["n_features"])
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        x_hold_t = torch.tensor(x_holdout, dtype=torch.float32)
        y_hold_t = torch.tensor(y_holdout, dtype=torch.float32)

        def build(ent: str) -> QuantumNetEntangled:
            return _build_model(input_dim, cfg, ent)

        if verbose:
            print(f"Seed {seed}: schedule...", flush=True)
        sched_acc = train_entangled_schedule(
            build,
            x_train_t,
            y_train_t,
            x_hold_t,
            y_hold_t,
            EXP_ID,
            f"entangle_schedule_seed{seed}",
            n_stages=n_stages,
            epochs_per_stage=epochs_per_stage,
            ladder=ladder,
            lr=lr,
            seed=seed,
            profile=profile,
        )
        schedule_accs.append(sched_acc)

        for topo in FIXED_TOPOLOGIES:
            if verbose:
                print(f"Seed {seed}: fixed {topo}...", flush=True)
            acc = train_fixed_entangled(
                build,
                topo,
                x_train_t,
                y_train_t,
                x_hold_t,
                y_hold_t,
                EXP_ID,
                f"entangle_{topo}_seed{seed}",
                n_stages=n_stages,
                epochs_per_stage=epochs_per_stage,
                lr=lr,
                seed=seed,
                profile=profile,
            )
            fixed_by_topo[topo].append(acc)

        if verbose:
            best_seed = max(FIXED_TOPOLOGIES, key=lambda t: fixed_by_topo[t][-1])
            print(
                f"Seed {seed}: schedule={sched_acc:.4f} | "
                f"best fixed ({best_seed})={fixed_by_topo[best_seed][-1]:.4f}",
                flush=True,
            )

    mean_by_topo = {t: statistics.mean(fixed_by_topo[t]) for t in FIXED_TOPOLOGIES}
    best_topo = max(mean_by_topo, key=mean_by_topo.get)
    best_fixed_mean = mean_by_topo[best_topo]
    mean_schedule = statistics.mean(schedule_accs)
    advantage_pp = (mean_schedule - best_fixed_mean) * 100.0
    best_fixed_accs = fixed_by_topo[best_topo]
    paired_wins = sum(1 for s, f in zip(schedule_accs, best_fixed_accs) if s > f)
    elapsed = time.perf_counter() - t0

    comparison = compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "entangle_schedule",
                "label_b": f"entangle_{best_topo}",
                "condition_a": schedule_accs,
                "condition_b": best_fixed_accs,
            }
        ],
    )
    wilcoxon_p = comparison[0].get("p_value") if comparison else None

    n_train = len(y_train) if seeds else 0
    n_holdout = len(y_holdout) if seeds else 0

    log_event(
        "info",
        "exp_053 entangle schedule summary",
        exp_id=EXP_ID,
        profile=profile,
        n_seeds=len(seeds),
        mean_schedule=round(mean_schedule, 4),
        best_fixed_topology=best_topo,
        best_fixed_mean=round(best_fixed_mean, 4),
        advantage_pp=round(advantage_pp, 3),
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
    )

    result = EntangleScheduleResult(
        n_seeds=len(seeds),
        n_train_rows=n_train,
        n_holdout_rows=n_holdout,
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        mean_schedule=mean_schedule,
        mean_by_topology=mean_by_topo,
        best_fixed_topology=best_topo,
        best_fixed_mean=best_fixed_mean,
        advantage_pp=advantage_pp,
        min_advantage_pp=min_advantage_pp,
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
        schedule_accuracies=tuple(schedule_accs),
        best_fixed_accuracies=tuple(best_fixed_accs),
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


def _summarize(result: EntangleScheduleResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    topo_lines = "\n".join(
        f"  {t}: {result.mean_by_topology[t] * 100:.2f}%"
        for t in FIXED_TOPOLOGIES
    )
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 053 SUMMARY",
            f"{'=' * 60}",
            f"Seeds: {result.n_seeds}",
            f"Stages: {result.n_stages} × {result.epochs_per_stage} epochs",
            f"Mean schedule holdout: {result.mean_schedule * 100:.2f}%",
            "Fixed topologies:",
            topo_lines,
            f"Best fixed: {result.best_fixed_topology} ({result.best_fixed_mean * 100:.2f}%)",
            f"Advantage: {result.advantage_pp:+.2f} pp (gate ≥ {result.min_advantage_pp} pp)",
            f"Paired wins: {result.paired_wins}/{result.n_seeds}",
            f"Wilcoxon p: {result.wilcoxon_p}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: EntangleScheduleResult) -> str:
    from datetime import date

    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 053: Dynamic entanglement schedule (breast cancer)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "**Dataset:** Wisconsin breast cancer (UCI), stratified holdout",
            "",
            "## Validation gate",
            "",
            f"- Seeds: **{result.n_seeds}**",
            f"- Train rows: **{result.n_train_rows}** · Holdout rows: **{result.n_holdout_rows}**",
            f"- Schedule stages: **{result.n_stages}** × **{result.epochs_per_stage}** epochs",
            f"- Mean schedule holdout: **{result.mean_schedule * 100:.2f}%**",
            f"- Best fixed ({result.best_fixed_topology}): **{result.best_fixed_mean * 100:.2f}%**",
            f"- Advantage: **{result.advantage_pp:+.2f} pp**",
            f"- Paired wins: **{result.paired_wins}/{result.n_seeds}**",
            f"- Wilcoxon p: **{result.wilcoxon_p}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — dynamic entanglement vs best fixed topology "
            f"(gate ≥ {result.min_advantage_pp} pp).",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 053 — dynamic entanglement schedule on breast cancer")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_053(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
