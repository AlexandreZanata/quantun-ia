"""
EXP 072 — Quantum warm-start vs end-to-end HybridSandwich on NIHR (C2 replication).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_072_quantum_warmstart_nihr/run.py --profile publication --write-results
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

from src.application.balanced_metrics import pr_auc
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.hybrid_model import HybridSandwich
from src.training.batched_trainer import train_model_batched
from src.training.config import load_experiment_config
from src.training.holdout import compare_conditions_batch
from src.training.quantum_warmstart import WarmStartConfig, train_hybrid_warmstart
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import predict

EXP_KEY = "exp_072_quantum_warmstart_nihr"
EXP_ID = "exp_072"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WarmStartNihrResult:
    n_seeds: int
    n_train_rows: int
    n_val_rows: int
    mean_e2e_pr_auc: float
    mean_warmstart_pr_auc: float
    advantage_pp: float
    min_vs_e2e_pp: float
    classical_epochs: int
    quantum_epochs: int
    total_epochs: int
    paired_wins: int
    wilcoxon_p: float | None
    e2e_pr_aucs: tuple[float, ...]
    warmstart_pr_aucs: tuple[float, ...]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _val_pr_auc(model: torch.nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = predict(model, x_val.to(device)).detach().cpu().numpy()
    labels = y_val.detach().cpu().numpy()
    score = pr_auc(labels, probs)
    return float(score) if score is not None else 0.5


def _build_hybrid(input_dim: int, cfg: dict) -> HybridSandwich:
    return HybridSandwich(
        input_dim=input_dim,
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
    )


def _train_kwargs(cfg: dict, *, seed: int, profile: str) -> dict:
    return {
        "lr": float(cfg.get("learning_rate", 0.01)),
        "batch_size": int(cfg.get("batch_size", 512)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "seed": seed,
        "profile": profile,
        "save_checkpoints": bool(cfg.get("save_checkpoints", False)),
    }


def _run_e2e(
    *,
    cfg: dict,
    input_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
) -> float:
    model = _build_hybrid(input_dim, cfg)
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        f"hybrid_e2e_seed{seed}",
        epochs=int(cfg["epochs"]),
        X_val=x_val,
        y_val=y_val,
        **_train_kwargs(cfg, seed=seed, profile=profile),
    )
    return _val_pr_auc(model, x_val, y_val)


def _run_warmstart(
    *,
    cfg: dict,
    input_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
) -> float:
    model = _build_hybrid(input_dim, cfg)
    ws_cfg = WarmStartConfig(
        classical_fraction=float(cfg.get("classical_fraction", 0.7)),
        total_epochs=int(cfg["epochs"]),
    )
    train_hybrid_warmstart(
        model,
        x_train,
        y_train,
        EXP_ID,
        f"hybrid_warmstart_seed{seed}",
        config=ws_cfg,
        x_val=x_val,
        y_val=y_val,
        **_train_kwargs(cfg, seed=seed, profile=profile),
    )
    return _val_pr_auc(model, x_val, y_val)


def gate_passed(result: WarmStartNihrResult) -> bool:
    return result.advantage_pp >= result.min_vs_e2e_pp


def run_exp_072(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> WarmStartNihrResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "nihr_cv_synthetic_v1"))
    seeds = tuple(int(s) for s in cfg.get("seeds", [42]))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 70_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 15_000)
    min_vs_e2e_pp = float(cfg.get("min_vs_e2e_pp", 0.5))
    classical_fraction = float(cfg.get("classical_fraction", 0.7))
    total_epochs = int(cfg["epochs"])
    classical_epochs = max(1, int(total_epochs * classical_fraction))
    quantum_epochs = max(1, total_epochs - classical_epochs)

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 072 — Quantum warm-start vs e2e hybrid (NIHR) | profile={profile} | "
            f"seeds={len(seeds)} | train={n_train or 'all'}"
        )
        print(
            f"Schedule: {classical_epochs} classical + {quantum_epochs} quantum epochs "
            f"({classical_fraction:.0%} classical-first)"
        )
        print(f"Gate: warm-start PR-AUC ≥ e2e + {min_vs_e2e_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
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

    e2e_scores: list[float] = []
    warmstart_scores: list[float] = []
    for seed in seeds:
        if verbose:
            print(f"Seed {seed}: e2e hybrid...", flush=True)
        e2e_score = _run_e2e(
            cfg=cfg,
            input_dim=input_dim,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            seed=seed,
            profile=profile,
        )
        e2e_scores.append(e2e_score)
        if verbose:
            print(f"Seed {seed}: warm-start (e2e PR-AUC={e2e_score:.4f})...", flush=True)
        ws_score = _run_warmstart(
            cfg=cfg,
            input_dim=input_dim,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            seed=seed,
            profile=profile,
        )
        warmstart_scores.append(ws_score)
        if verbose:
            print(
                f"Seed {seed}: e2e={e2e_score:.4f} warm-start={ws_score:.4f} "
                f"Δ={(ws_score - e2e_score) * 100:.2f} pp",
                flush=True,
            )

    mean_e2e = statistics.mean(e2e_scores)
    mean_ws = statistics.mean(warmstart_scores)
    advantage_pp = (mean_ws - mean_e2e) * 100.0
    paired_wins = sum(1 for e, w in zip(e2e_scores, warmstart_scores) if w > e)
    elapsed = time.perf_counter() - t0

    comparison = compare_conditions_batch(
        EXP_ID,
        [
            {
                "label_a": "hybrid_warmstart",
                "label_b": "hybrid_e2e",
                "condition_a": warmstart_scores,
                "condition_b": e2e_scores,
            }
        ],
    )
    wilcoxon_p = comparison[0].get("p_value") if comparison else None

    log_event(
        "info",
        "exp_072 warmstart summary",
        exp_id=EXP_ID,
        profile=profile,
        n_seeds=len(seeds),
        mean_e2e_pr_auc=round(mean_e2e, 4),
        mean_warmstart_pr_auc=round(mean_ws, 4),
        advantage_pp=round(advantage_pp, 3),
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
        elapsed_s=round(elapsed, 3),
    )

    result = WarmStartNihrResult(
        n_seeds=len(seeds),
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        mean_e2e_pr_auc=mean_e2e,
        mean_warmstart_pr_auc=mean_ws,
        advantage_pp=advantage_pp,
        min_vs_e2e_pp=min_vs_e2e_pp,
        classical_epochs=classical_epochs,
        quantum_epochs=quantum_epochs,
        total_epochs=total_epochs,
        paired_wins=paired_wins,
        wilcoxon_p=wilcoxon_p,
        e2e_pr_aucs=tuple(e2e_scores),
        warmstart_pr_aucs=tuple(warmstart_scores),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"\nMean e2e={mean_e2e:.4f} | warm-start={mean_ws:.4f} | "
            f"Δ={advantage_pp:.2f} pp | wins={paired_wins}/{len(seeds)} [{status}] | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return result


def _summarize(result: WarmStartNihrResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 072 SUMMARY",
            f"{'=' * 60}",
            f"Seeds: {result.n_seeds}",
            f"Schedule: {result.classical_epochs} classical + {result.quantum_epochs} quantum "
            f"({result.total_epochs} total)",
            f"Mean e2e PR-AUC: {result.mean_e2e_pr_auc:.4f}",
            f"Mean warm-start PR-AUC: {result.mean_warmstart_pr_auc:.4f}",
            f"Advantage: {result.advantage_pp:+.2f} pp (gate ≥ {result.min_vs_e2e_pp} pp)",
            f"Paired wins: {result.paired_wins}/{result.n_seeds}",
            f"Wilcoxon p: {result.wilcoxon_p}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: WarmStartNihrResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 072: Quantum warm-start on NIHR hybrid (C2 replication)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (PR-AUC)",
            "",
            f"- Seeds: **{result.n_seeds}**",
            f"- Train rows: **{result.n_train_rows:,}** · Val rows: **{result.n_val_rows:,}**",
            f"- Schedule: **{result.classical_epochs}** classical + **{result.quantum_epochs}** quantum epochs",
            f"- Mean e2e PR-AUC: **{result.mean_e2e_pr_auc:.4f}**",
            f"- Mean warm-start PR-AUC: **{result.mean_warmstart_pr_auc:.4f}**",
            f"- Advantage: **{result.advantage_pp:+.2f} pp**",
            f"- Paired wins: **{result.paired_wins}/{result.n_seeds}**",
            f"- Wilcoxon p: **{result.wilcoxon_p}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — warm-start val PR-AUC vs end-to-end hybrid "
            f"(gate ≥ {result.min_vs_e2e_pp} pp).",
            "",
            "## Limitations",
            "- HybridSandwich protocol (mirrors exp_052); not frozen C2 backbone.",
            "- QNN sim on CPU; val PR-AUC only.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 072 — quantum warm-start vs e2e hybrid (NIHR)")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_072(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
