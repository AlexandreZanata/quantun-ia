"""
EXP 031 — Clinical curriculum ablation: margin_batches vs random on breast cancer.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_031_curriculum_clinical/run.py --profile ci
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
from src.quantum.hybrid_model import HybridSandwich
from src.training.config import load_experiment_config
from src.training.curriculum import (
    curriculum_total_epochs,
    sort_by_difficulty,
    train_curriculum_batched,
)
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.protocol import log_applicability_gate, log_experiment_protocol, task_learnable
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import evaluate, train_model

EXP_KEY = "exp_031_curriculum_clinical"
EXP_ID = "exp_031"
MODEL = "hybrid_sandwich"


@dataclass(frozen=True)
class CurriculumClinicalResult:
    n_seeds: int
    mean_random: float
    mean_curriculum: float
    advantage_pp: float
    min_advantage_pp: float
    paired_wins: int
    applicable: bool
    random_accuracies: tuple[float, ...]
    curriculum_accuracies: tuple[float, ...]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _build_hybrid(input_dim: int, cfg: dict) -> HybridSandwich:
    mc = cfg.get("model_configs", {}).get(MODEL, {})
    return HybridSandwich(
        input_dim=input_dim,
        n_qubits=int(mc.get("n_qubits", cfg.get("n_qubits", 4))),
        n_layers=int(mc.get("n_layers", cfg.get("n_layers", 2))),
        reupload=bool(mc.get("reupload", True)),
    )


def _run_random_baseline(
    *,
    cfg: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
    input_dim: int,
) -> float:
    lr = float(cfg.get("model_configs", {}).get(MODEL, {}).get("learning_rate", cfg.get("learning_rate", 0.02)))
    epochs = curriculum_total_epochs(cfg)
    model = _build_hybrid(input_dim, cfg)
    X_sorted, y_sorted = sort_by_difficulty(X_train, y_train, method="random")
    train_model(
        model,
        torch.tensor(X_sorted),
        torch.tensor(y_sorted),
        EXP_ID,
        f"curriculum_random_seed{seed}",
        epochs=epochs,
        lr=lr,
        X_test=torch.tensor(X_test),
        y_test=torch.tensor(y_test),
        seed=seed,
        profile=cfg.get("profile"),
        save_checkpoints=False,
    )
    return float(evaluate(model, torch.tensor(X_test), torch.tensor(y_test))["accuracy"])


def _run_margin_curriculum(
    *,
    cfg: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
    input_dim: int,
) -> float:
    lr = float(cfg.get("model_configs", {}).get(MODEL, {}).get("learning_rate", cfg.get("learning_rate", 0.02)))
    model = _build_hybrid(input_dim, cfg)
    result = train_curriculum_batched(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        exp_id=EXP_ID,
        model_name=f"curriculum_margin_batches_seed{seed}",
        n_stages=int(cfg["curriculum_stages"]),
        epochs_per_stage=int(cfg["epochs_per_stage"]),
        lr=lr,
        refine_epochs=int(cfg.get("refine_epochs", 12)),
    )
    return float(result["test_accuracy"])


def run_exp_031(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> CurriculumClinicalResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds: list[int] = list(cfg["seeds"])
    threshold = float(cfg.get("learnability_threshold", 0.90))
    min_advantage_pp = float(cfg.get("min_advantage_pp", 0.0))
    test_size = float(cfg.get("test_size", 0.3))
    dataset = str(cfg.get("dataset", "breast_cancer"))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 031 — Clinical Curriculum | profile={profile} | seeds={len(seeds)}")
        print(f"Dataset: {dataset} | gate: advantage > {min_advantage_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    random_accs: list[float] = []
    curriculum_accs: list[float] = []

    for i, seed in enumerate(seeds, start=1):
        X_train, X_test, y_train, y_test, _meta = prepare_dataset(
            dataset, random_state=seed, test_size=test_size
        )
        input_dim = X_train.shape[1]

        if verbose:
            print(f"[{i}/{len(seeds)}] seed={seed} — random baseline...", flush=True)
        r_acc = _run_random_baseline(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            input_dim=input_dim,
        )
        random_accs.append(r_acc)
        if verbose:
            print(f"  random holdout={r_acc * 100:.2f}%", flush=True)

        if verbose:
            print(f"[{i}/{len(seeds)}] seed={seed} — margin curriculum...", flush=True)
        c_acc = _run_margin_curriculum(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            input_dim=input_dim,
        )
        curriculum_accs.append(c_acc)
        if verbose:
            delta = (c_acc - r_acc) * 100
            print(f"  curriculum holdout={c_acc * 100:.2f}% (Δ={delta:+.2f} pp)", flush=True)

    elapsed = time.perf_counter() - t0
    mean_random = statistics.mean(random_accs)
    mean_curriculum = statistics.mean(curriculum_accs)
    advantage_pp = (mean_curriculum - mean_random) * 100.0
    paired_wins = sum(c > r for c, r in zip(curriculum_accs, random_accs, strict=True))
    applicable = task_learnable(random_accs, threshold)

    log_applicability_gate(
        EXP_ID,
        "curriculum",
        applicable,
        threshold=threshold,
        mean_holdout=mean_random,
        reason="hybrid_sandwich random baseline on breast cancer vs learnability threshold",
    )

    summarize_multi_seed(
        EXP_ID,
        {"curriculum_random": random_accs, "curriculum_margin_batches": curriculum_accs},
    )
    compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "margin_batches",
                "label_b": "random",
                "condition_a": curriculum_accs,
                "condition_b": random_accs,
            },
        ],
    )

    if verbose:
        status = "OK" if applicable and advantage_pp > min_advantage_pp else "FAIL"
        print(
            f"\nmean_random={mean_random * 100:.2f}% mean_curriculum={mean_curriculum * 100:.2f}% "
            f"advantage={advantage_pp:+.2f} pp wins={paired_wins}/{len(seeds)} [{status}] "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return CurriculumClinicalResult(
        n_seeds=len(seeds),
        mean_random=mean_random,
        mean_curriculum=mean_curriculum,
        advantage_pp=advantage_pp,
        min_advantage_pp=min_advantage_pp,
        paired_wins=paired_wins,
        applicable=applicable,
        random_accuracies=tuple(random_accs),
        curriculum_accuracies=tuple(curriculum_accs),
        elapsed_s=round(elapsed, 3),
    )


def _summarize(result: CurriculumClinicalResult) -> str:
    accepted = result.applicable and result.advantage_pp > result.min_advantage_pp
    verdict = "accepted" if accepted else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 031 SUMMARY",
            f"{'=' * 60}",
            f"Seeds: {result.n_seeds}",
            f"Mean random: {result.mean_random * 100:.2f}%",
            f"Mean curriculum: {result.mean_curriculum * 100:.2f}%",
            f"Advantage: {result.advantage_pp:+.2f} pp (gate > {result.min_advantage_pp} pp)",
            f"Paired wins: {result.paired_wins}/{result.n_seeds}",
            f"Applicable: {result.applicable}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 031 — clinical curriculum ablation")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    init_correlation_id()
    cfg = load_experiment_config(EXP_KEY, profile=args.profile)
    log_experiment_protocol(EXP_ID, cfg)
    log_event("info", "experiment run started", exp_id=EXP_ID, profile=args.profile)

    result = run_exp_031(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))
    log_event("info", "experiment run finished", exp_id=EXP_ID, advantage_pp=result.advantage_pp)

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    accepted = result.applicable and result.advantage_pp > result.min_advantage_pp
    return 0 if accepted else 1


def _build_results_md(result: CurriculumClinicalResult) -> str:
    from datetime import date

    verdict = "accepted" if result.applicable and result.advantage_pp > result.min_advantage_pp else "rejected"
    return "\n".join(
        [
            "# Results — EXP 031: Clinical Curriculum Ablation",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "**Dataset:** breast_cancer (UCI Wisconsin), 30% holdout",
            "",
            "## Holdout comparison",
            "",
            f"| Method | Mean |",
            f"|--------|------|",
            f"| curriculum_random | **{result.mean_random * 100:.2f}%** |",
            f"| curriculum_margin_batches | **{result.mean_curriculum * 100:.2f}%** |",
            "",
            f"- Advantage: **{result.advantage_pp:+.2f} pp**",
            f"- Paired wins (curriculum > random): **{result.paired_wins}/{result.n_seeds}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — margin curriculum mean holdout vs epoch-matched random baseline.",
            "",
            "## Limitations",
            "- Single clinical dataset; not a deployment claim.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
