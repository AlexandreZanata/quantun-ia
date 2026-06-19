"""
EXP 046 — LargeNanoMLP scale curve on HIGGS (nano_s → nano_xxl).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_046_model_scale_curve/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc_batched, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_046_model_scale_curve"
EXP_ID = "exp_046"
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VARIANTS: dict[str, dict[str, int | float]] = {
    "nano_s": {"hidden1": 512, "hidden2": 128, "hidden3": 32, "batch_size": 4096},
    "nano_m": {"hidden1": 1024, "hidden2": 256, "hidden3": 64, "batch_size": 2048},
    "nano_l": {"hidden1": 2048, "hidden2": 512, "hidden3": 64, "batch_size": 2048},
    "nano_xl": {"hidden1": 4096, "hidden2": 1024, "hidden3": 128, "batch_size": 2048},
    "nano_xxl": {"hidden1": 4096, "hidden2": 2048, "hidden3": 256, "batch_size": 1024},
}


@dataclass(frozen=True)
class VariantResult:
    variant_key: str
    n_params: int
    val_roc_auc: float
    elapsed_s: float
    peak_vram_mb: float | None
    oom: bool
    skipped: bool
    hidden1: int
    hidden2: int
    hidden3: int
    batch_size: int


@dataclass(frozen=True)
class ModelScaleResult:
    n_train_rows: int
    n_val_rows: int
    variants: tuple[VariantResult, ...]
    min_xl_advantage_pp: float
    xl_advantage_pp: float | None


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)


def _reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _train_variant(
    *,
    variant_key: str,
    spec: dict[str, Any],
    input_dim: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict[str, Any],
    profile: str,
    seed: int,
) -> VariantResult:
    hidden1 = int(spec["hidden1"])
    hidden2 = int(spec["hidden2"])
    hidden3 = int(spec["hidden3"])
    batch_size = int(spec.get("batch_size", cfg.get("batch_size", 2048)))
    dropout = float(spec.get("dropout", cfg.get("dropout", 0.3)))

    _reset_vram_stats()
    t0 = time.perf_counter()
    oom = False
    val_auc = 0.0
    n_params = 0

    try:
        model = LargeNanoMLP(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            hidden3=hidden3,
            dropout=dropout,
        )
        n_params = count_parameters(model)
        train_model_batched(
            model,
            x_train,
            y_train,
            EXP_ID,
            f"large_nano_mlp_{variant_key}",
            epochs=int(cfg["epochs"]),
            lr=float(cfg.get("learning_rate", 0.001)),
            batch_size=batch_size,
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            X_val=x_val,
            y_val=y_val,
            seed=seed,
            profile=profile,
            save_checkpoints=False,
            device="cuda",
        )
        eval_batch = int(cfg.get("eval_batch_size", 8192))
        metrics = evaluate_with_auc_batched(model, x_val, y_val, batch_size=eval_batch)
        val_auc = float(metrics["roc_auc"])
    except torch.cuda.OutOfMemoryError:
        oom = True
        val_auc = 0.0
    finally:
        elapsed = time.perf_counter() - t0
        peak = _peak_vram_mb()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return VariantResult(
        variant_key=variant_key,
        n_params=n_params,
        val_roc_auc=val_auc,
        elapsed_s=round(elapsed, 3),
        peak_vram_mb=peak,
        oom=oom,
        skipped=False,
        hidden1=hidden1,
        hidden2=hidden2,
        hidden3=hidden3,
        batch_size=batch_size,
    )


def run_exp_046(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ModelScaleResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "higgs_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 805_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 172_500)
    min_xl_pp = float(cfg.get("min_xl_advantage_pp", 0.3))
    variant_specs: dict[str, dict[str, Any]] = dict(cfg.get("variants") or DEFAULT_VARIANTS)
    variant_order = list(cfg.get("variant_order") or variant_specs.keys())

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 046 — Model scale curve | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Variants: {', '.join(variant_order)}")
        print(f"Gate: nano_xl − nano_l ≥ {min_xl_pp} pp")
        print(f"{'=' * 60}\n")

    x_train, y_train, x_val, y_val, _x_test, _y_test, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    results: list[VariantResult] = []
    for key in variant_order:
        if key not in variant_specs:
            continue
        if verbose:
            print(f"--- {key} ---", flush=True)
        outcome = _train_variant(
            variant_key=key,
            spec=variant_specs[key],
            input_dim=input_dim,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            cfg=cfg,
            profile=profile,
            seed=seed,
        )
        results.append(outcome)
        log_event(
            "info",
            "exp_046 variant complete",
            exp_id=EXP_ID,
            profile=profile,
            variant_key=key,
            n_params=outcome.n_params,
            val_roc_auc=outcome.val_roc_auc,
            peak_vram_mb=outcome.peak_vram_mb,
            oom=outcome.oom,
            elapsed_s=outcome.elapsed_s,
        )
        if verbose:
            status = "OOM" if outcome.oom else f"AUC={outcome.val_roc_auc:.4f}"
            print(
                f"{key}: params={outcome.n_params:,} | {status} | "
                f"VRAM={outcome.peak_vram_mb} MB | {outcome.elapsed_s}s",
                flush=True,
            )

    by_key = {r.variant_key: r for r in results}
    xl_pp: float | None = None
    if "nano_xl" in by_key and "nano_l" in by_key and not by_key["nano_xl"].oom and not by_key["nano_l"].oom:
        xl_pp = (by_key["nano_xl"].val_roc_auc - by_key["nano_l"].val_roc_auc) * 100.0

    return ModelScaleResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        variants=tuple(results),
        min_xl_advantage_pp=min_xl_pp,
        xl_advantage_pp=round(xl_pp, 3) if xl_pp is not None else None,
    )


def gate_passed(result: ModelScaleResult) -> bool:
    if result.xl_advantage_pp is None:
        return False
    return result.xl_advantage_pp >= result.min_xl_advantage_pp


def _summarize(result: ModelScaleResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    lines = [
        f"\n{'=' * 60}",
        "EXP 046 SUMMARY",
        f"{'=' * 60}",
        f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
    ]
    for v in result.variants:
        tag = "OOM" if v.oom else f"AUC={v.val_roc_auc:.4f}"
        lines.append(
            f"  {v.variant_key}: {v.n_params:,} params | {tag} | "
            f"VRAM={v.peak_vram_mb} MB | {v.elapsed_s}s"
        )
    if result.xl_advantage_pp is not None:
        lines.append(
            f"nano_xl − nano_l: {result.xl_advantage_pp:.2f} pp "
            f"(gate ≥ {result.min_xl_advantage_pp} pp)"
        )
    lines.extend([f"Verdict: {verdict}", f"{'=' * 60}\n"])
    return "\n".join(lines)


def _build_results_md(result: ModelScaleResult) -> str:
    from datetime import date

    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    rows = [
        "# Results — EXP 046: Model scale curve on HIGGS",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "",
        "## Scale curve",
        "",
        "| Variant | Params | Val ROC-AUC | Peak VRAM (MB) | Wall time (s) | Status |",
        "|---------|--------|-------------|----------------|---------------|--------|",
    ]
    for v in result.variants:
        status = "OOM" if v.oom else "ok"
        auc = f"{v.val_roc_auc:.4f}" if not v.oom else "—"
        rows.append(
            f"| {v.variant_key} | {v.n_params:,} | {auc} | {v.peak_vram_mb} | "
            f"{v.elapsed_s} | {status} |"
        )
    rows.extend(
        [
            "",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
        ]
    )
    if result.xl_advantage_pp is not None:
        rows.append(
            f"- nano_xl − nano_l: **{result.xl_advantage_pp:.2f} pp** "
            f"(gate ≥ {result.min_xl_advantage_pp} pp)"
        )
    rows.extend(
        [
            "",
            "## Verdict",
            f"**{verdict}**",
            "",
            "## Limitations",
            "- Single seed; multi-seed Wilcoxon deferred to overnight profile (Phase 1.2).",
            "- Val-only metrics; test split untouched.",
            "",
        ]
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 046 — LargeNanoMLP scale curve on HIGGS")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_046(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
