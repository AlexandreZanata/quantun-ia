"""
EXP 044 — LargeNanoMLP on NIHR synthetic CV vs logistic baseline.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_044_nihr_cv_baseline/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.logistic_baseline import LogisticBaseline
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc, evaluate_with_auc_batched, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_044_nihr_cv_baseline"
EXP_ID = "exp_044"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NihrBaselineResult:
    n_params: int
    n_train_rows: int
    n_val_rows: int
    logistic_val_auc: float
    nano_val_auc: float
    auc_advantage_pp: float
    min_val_auc: float
    min_auc_advantage_pp: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def run_exp_044(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> NihrBaselineResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "nihr_cv_synthetic_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 70_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 15_000)
    min_val_auc = float(cfg.get("min_val_auc", 0.70))
    min_auc_pp = float(cfg.get("min_auc_advantage_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 044 — NIHR CV baseline | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: val AUC ≥ {min_val_auc} | nano − logistic ≥ {min_auc_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
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

    logistic = LogisticBaseline(input_dim, random_state=seed)
    logistic.train(
        x_train_t,
        y_train_t,
        EXP_ID,
        "logistic_baseline",
        X_test=x_val_t,
        y_test=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    logistic_metrics = evaluate_with_auc(logistic, x_val_t, y_val_t)
    logistic_auc = float(logistic_metrics["roc_auc"])

    model = LargeNanoMLP(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    n_params = count_parameters(model)
    train_model_batched(
        model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "large_nano_mlp",
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
        device="cuda",
    )
    nano_metrics = evaluate_with_auc_batched(model, x_val_t, y_val_t)
    nano_auc = float(nano_metrics["roc_auc"])
    advantage_pp = (nano_auc - logistic_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_044 complete",
        exp_id=EXP_ID,
        profile=profile,
        n_params=n_params,
        logistic_val_auc=logistic_auc,
        nano_val_auc=nano_auc,
        auc_advantage_pp=round(advantage_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        print(
            f"logistic AUC={logistic_auc:.4f} | nano AUC={nano_auc:.4f} | "
            f"Δ={advantage_pp:.2f} pp | params={n_params:,} | {elapsed:.1f}s",
            flush=True,
        )

    return NihrBaselineResult(
        n_params=n_params,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        logistic_val_auc=logistic_auc,
        nano_val_auc=nano_auc,
        auc_advantage_pp=round(advantage_pp, 3),
        min_val_auc=min_val_auc,
        min_auc_advantage_pp=min_auc_pp,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: NihrBaselineResult) -> bool:
    return (
        result.nano_val_auc >= result.min_val_auc
        and result.auc_advantage_pp >= result.min_auc_advantage_pp
    )


def _summarize(result: NihrBaselineResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 044 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Logistic val AUC: {result.logistic_val_auc:.4f}",
            f"LargeNanoMLP val AUC: {result.nano_val_auc:.4f} ({result.n_params:,} params)",
            f"nano − logistic: {result.auc_advantage_pp:.2f} pp "
            f"(gate ≥ {result.min_auc_advantage_pp} pp)",
            f"Val AUC gate: {result.nano_val_auc:.4f} (need ≥ {result.min_val_auc})",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: NihrBaselineResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    return "\n".join(
        [
            "# Results — EXP 044: NIHR synthetic CV baseline",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Holdout metrics",
            "",
            f"| Model | Val ROC-AUC |",
            f"|-------|-------------|",
            f"| Logistic (QRISK-style) | {result.logistic_val_auc:.4f} |",
            f"| LargeNanoMLP | {result.nano_val_auc:.4f} |",
            "",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Params: **{result.n_params:,}**",
            f"- nano − logistic: **{result.auc_advantage_pp:.2f} pp**",
            f"- Wall time: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}**",
            "",
            "## Limitations",
            "- Synthetic NIHR cohort (CC0 Zenodo) — not clinical deployment.",
            "- Train-only median imputation; calibrated head deferred to exp_047.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 044 — NIHR CV baseline")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_044(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
