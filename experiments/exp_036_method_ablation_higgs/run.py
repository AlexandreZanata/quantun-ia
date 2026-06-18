"""
EXP 036 — Training methodology ablation on HIGGS (LargeNanoMLP).

Compare baseline vs curriculum vs adaptive LR vs champion loop on val ROC-AUC.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_036_method_ablation_higgs/run.py --profile ci
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.adaptive_lr import AdaptiveLRConfig, compute_lr_scale, step_gradient_variance
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.curriculum import curriculum_batches
from src.training.holdout import compare_conditions_batch, summarize_multi_seed
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_036_method_ablation_higgs"
EXP_ID = "exp_036"
ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "curriculum", "adaptive", "champion")


@dataclass(frozen=True)
class MethodAblationResult:
    n_seeds: int
    n_train_rows: int
    n_val_rows: int
    mean_auc_by_method: dict[str, float]
    baseline_mean_auc: float
    best_method: str
    best_advantage_pp: float
    min_beat_baseline_pp: float
    beaters: tuple[str, ...]
    auc_by_method_seed: dict[str, tuple[float, ...]]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _build_model(input_dim: int, cfg: dict) -> LargeNanoMLP:
    return LargeNanoMLP(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
    )


def _train_kwargs(cfg: dict, *, seed: int, profile: str) -> dict:
    return {
        "lr": float(cfg.get("learning_rate", 0.001)),
        "batch_size": int(cfg.get("batch_size", 1024)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "seed": seed,
        "profile": profile,
        "save_checkpoints": False,
    }


def _eval_val_auc(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    return float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])


def _run_baseline(
    *,
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
    input_dim: int,
) -> float:
    model = _build_model(input_dim, cfg)
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        f"baseline_seed{seed}",
        epochs=int(cfg.get("baseline_epochs", 20)),
        X_val=x_val,
        y_val=y_val,
        **_train_kwargs(cfg, seed=seed, profile=profile),
    )
    return _eval_val_auc(model, x_val, y_val)


def _run_curriculum(
    *,
    cfg: dict,
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
    input_dim: int,
) -> float:
    model = _build_model(input_dim, cfg)
    n_stages = int(cfg.get("curriculum_stages", 4))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 12))
    refine_epochs = int(cfg.get("refine_epochs", 12))
    kwargs = _train_kwargs(cfg, seed=seed, profile=profile)

    for stage_idx, (x_stage, y_stage) in enumerate(curriculum_batches(x_train_np, y_train_np, n_stages)):
        train_model_batched(
            model,
            torch.tensor(x_stage, dtype=torch.float32),
            torch.tensor(y_stage, dtype=torch.float32),
            EXP_ID,
            f"curriculum_stage{stage_idx}_seed{seed}",
            epochs=epochs_per_stage,
            X_val=x_val,
            y_val=y_val,
            **kwargs,
        )

    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        f"curriculum_refine_seed{seed}",
        epochs=refine_epochs,
        X_val=x_val,
        y_val=y_val,
        **kwargs,
    )
    return _eval_val_auc(model, x_val, y_val)


def _train_adaptive_batched(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    exp_id: str,
    model_name: str,
    epochs: int,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    seed: int,
    profile: str,
) -> None:
    """Mini-batch training with gradient-variance LR scaling (RTX 4060 safe)."""
    from src.training.device import resolve_device
    from src.training.reproducibility import set_global_seed

    adapt_cfg = AdaptiveLRConfig(
        base_lr=float(cfg.get("learning_rate", 0.001)),
        var_target=float(cfg.get("adaptive_var_target", 0.015)),
    )
    if seed is not None:
        set_global_seed(seed)

    dev = resolve_device(None, model=model)
    model = model.to(dev)
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_val = x_val.to(dev)
    y_val = y_val.to(dev)

    batch_size = int(cfg.get("batch_size", 1024))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    current_lr = adapt_cfg.base_lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=current_lr,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        if epoch >= adapt_cfg.warmup_epochs and (epoch % adapt_cfg.adapt_every == 0):
            x_sample, y_sample = next(iter(loader))
            grad_var = step_gradient_variance(model, x_sample, y_sample, criterion)
            scale = compute_lr_scale(grad_var, adapt_cfg)
            current_lr = adapt_cfg.base_lr * scale
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        if epoch == epochs - 1:
            _ = evaluate_with_auc(model, x_val, y_val)


def _run_adaptive(
    *,
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
    input_dim: int,
) -> float:
    model = _build_model(input_dim, cfg)
    _train_adaptive_batched(
        model,
        x_train,
        y_train,
        exp_id=EXP_ID,
        model_name=f"adaptive_seed{seed}",
        epochs=int(cfg.get("adaptive_epochs", 50)),
        x_val=x_val,
        y_val=y_val,
        cfg=cfg,
        seed=seed,
        profile=profile,
    )
    return _eval_val_auc(model, x_val, y_val)


def _run_champion(
    *,
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
    input_dim: int,
) -> float:
    """Two-cycle retrain proxy — keep best val AUC (champion promotion simplified)."""
    epochs = int(cfg.get("champion_epochs", 30))
    cycle_epochs = max(epochs // 2, 1)
    kwargs = _train_kwargs(cfg, seed=seed, profile=profile)
    best_auc = 0.0

    for cycle in range(2):
        cycle_seed = seed + cycle * 17
        model = _build_model(input_dim, cfg)
        train_model_batched(
            model,
            x_train,
            y_train,
            EXP_ID,
            f"champion_cycle{cycle}_seed{seed}",
            epochs=cycle_epochs,
            X_val=x_val,
            y_val=y_val,
            seed=cycle_seed,
            profile=profile,
            lr=kwargs["lr"] * (0.9 if cycle else 1.0),
            batch_size=kwargs["batch_size"],
            weight_decay=kwargs["weight_decay"],
            save_checkpoints=False,
        )
        best_auc = max(best_auc, _eval_val_auc(model, x_val, y_val))

    return best_auc


def _run_method(
    method: str,
    *,
    cfg: dict,
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    seed: int,
    profile: str,
    input_dim: int,
) -> float:
    runners = {
        "baseline": _run_baseline,
        "curriculum": _run_curriculum,
        "adaptive": _run_adaptive,
        "champion": _run_champion,
    }
    runner = runners[method]
    if method == "curriculum":
        return runner(
            cfg=cfg,
            x_train_np=x_train_np,
            y_train_np=y_train_np,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=seed,
            profile=profile,
            input_dim=input_dim,
        )
    return runner(
        cfg=cfg,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        input_dim=input_dim,
    )


def run_exp_036(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> MethodAblationResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds: list[int] = list(cfg["seeds"])
    dataset_id = str(cfg.get("dataset_id", "higgs_v1"))
    seed_base = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 805_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 172_500)
    min_beat_pp = float(cfg.get("min_beat_baseline_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXP 036 — Method Ablation HIGGS | profile={profile} | seeds={len(seeds)}")
        print(f"Methods: {', '.join(METHODS)} | gate: beat baseline ≥ {min_beat_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    auc_by_method: dict[str, list[float]] = {m: [] for m in METHODS}

    for i, seed in enumerate(seeds, start=1):
        x_train, y_train, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
            dataset_id,
            ROOT,
            n_train_rows=n_train,
            n_val_rows=n_val,
            random_state=seed_base + seed,
        )
        input_dim = int(x_train.shape[1])
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        x_val_t = torch.tensor(x_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        if verbose:
            print(f"[{i}/{len(seeds)}] seed={seed} train={len(y_train):,} val={len(y_val):,}", flush=True)

        for method in METHODS:
            if verbose:
                print(f"  {method}...", flush=True)
            auc = _run_method(
                method,
                cfg=cfg,
                x_train_np=x_train,
                y_train_np=y_train,
                x_train=x_train_t,
                y_train=y_train_t,
                x_val=x_val_t,
                y_val=y_val_t,
                seed=seed,
                profile=profile,
                input_dim=input_dim,
            )
            auc_by_method[method].append(auc)
            if verbose:
                print(f"    val_auc={auc:.4f}", flush=True)

    elapsed = time.perf_counter() - t0
    mean_by_method = {m: statistics.mean(v) for m, v in auc_by_method.items()}
    baseline_mean = mean_by_method["baseline"]
    advantages = {
        m: (mean_by_method[m] - baseline_mean) * 100.0
        for m in METHODS
        if m != "baseline"
    }
    beaters = tuple(m for m, adv in advantages.items() if adv >= min_beat_pp)
    best_method = max(advantages, key=advantages.get) if advantages else "baseline"
    best_advantage_pp = advantages.get(best_method, 0.0)

    summarize_multi_seed(
        EXP_ID,
        {f"{m}_val_auc": v for m, v in auc_by_method.items()},
    )
    comparisons = [
        {
            "label_a": alt,
            "label_b": "baseline",
            "condition_a": auc_by_method[alt],
            "condition_b": auc_by_method["baseline"],
        }
        for alt in METHODS
        if alt != "baseline"
    ]
    compare_conditions_batch(EXP_ID, comparisons)

    log_event(
        "info",
        "exp_036 ablation summary",
        exp_id=EXP_ID,
        profile=profile,
        baseline_mean_auc=baseline_mean,
        best_method=best_method,
        best_advantage_pp=round(best_advantage_pp, 3),
        beaters=list(beaters),
    )

    if verbose:
        print(
            f"\nbaseline={baseline_mean:.4f} | best={best_method} ({best_advantage_pp:+.2f} pp) | "
            f"beaters={list(beaters) or 'none'} | elapsed={elapsed:.1f}s",
            flush=True,
        )

    n_train_actual = len(y_train) if seeds else 0
    n_val_actual = len(y_val) if seeds else 0

    return MethodAblationResult(
        n_seeds=len(seeds),
        n_train_rows=n_train_actual,
        n_val_rows=n_val_actual,
        mean_auc_by_method=mean_by_method,
        baseline_mean_auc=baseline_mean,
        best_method=best_method,
        best_advantage_pp=best_advantage_pp,
        min_beat_baseline_pp=min_beat_pp,
        beaters=beaters,
        auc_by_method_seed={m: tuple(v) for m, v in auc_by_method.items()},
        elapsed_s=round(elapsed, 3),
    )


def _passed(result: MethodAblationResult) -> bool:
    return len(result.beaters) > 0


def _summarize(result: MethodAblationResult) -> str:
    verdict = "accepted" if _passed(result) else "rejected (honest negative)"
    lines = [
        f"\n{'=' * 60}",
        "EXP 036 SUMMARY",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 036 — HIGGS methodology ablation")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_036(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0


def _build_results_md(result: MethodAblationResult) -> str:
    from datetime import date

    verdict = "accepted" if _passed(result) else "rejected (honest negative)"
    rows = [
        "# Results — EXP 036: HIGGS Methodology Ablation",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Mean val ROC-AUC by method",
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
            f"- Seeds: **{result.n_seeds}**",
            f"- Beaters (≥ {result.min_beat_baseline_pp} pp): **{list(result.beaters) or 'none'}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — paired comparison vs Adam baseline on HIGGS slice.",
            "",
        ]
    )
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
