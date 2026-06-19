"""
EXP 060 — LargeNanoMLP on ACYD Brazil soybean vs logistic baseline (C4 anchor).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_060_large_nano_acyd_soy/run.py --profile publication
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.logistic_baseline import LogisticBaseline
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_060_large_nano_acyd_soy"
EXP_ID = "exp_060"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LargeNanoAcydResult:
    n_params: int
    n_train_rows: int
    n_val_rows: int
    logistic_val_auc: float
    nano_val_auc: float
    auc_advantage_pp: float
    min_auc_advantage_pp: float
    min_params: int
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def run_exp_060(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> LargeNanoAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 501_07)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    min_auc_pp = float(cfg.get("min_auc_advantage_pp", 2.0))
    min_params = int(cfg.get("min_params", 1_000_000))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 060 — LargeNanoMLP on ACYD soybean | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: val AUC ≥ logistic + {min_auc_pp} pp | params ≥ {min_params:,}")
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

    logistic = LogisticBaseline(input_dim=input_dim, random_state=seed)
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
    if n_params < min_params:
        raise RuntimeError(f"LargeNanoMLP has {n_params:,} params < {min_params:,} floor")

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
    )
    nano_metrics = evaluate_with_auc(model, x_val_t, y_val_t)
    nano_auc = float(nano_metrics["roc_auc"])
    advantage_pp = (nano_auc - logistic_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_060 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        dataset_id=dataset_id,
        n_params=n_params,
        logistic_val_auc=logistic_auc,
        nano_val_auc=nano_auc,
        auc_advantage_pp=round(advantage_pp, 3),
    )

    if verbose:
        status = "OK" if advantage_pp >= min_auc_pp else "FAIL"
        print(
            f"logistic AUC={logistic_auc:.4f} | nano AUC={nano_auc:.4f} | "
            f"Δ={advantage_pp:.2f} pp [{status}] | params={n_params:,} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return LargeNanoAcydResult(
        n_params=n_params,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        logistic_val_auc=logistic_auc,
        nano_val_auc=nano_auc,
        auc_advantage_pp=advantage_pp,
        min_auc_advantage_pp=min_auc_pp,
        min_params=min_params,
        elapsed_s=round(elapsed, 3),
    )


def _summarize(result: LargeNanoAcydResult) -> str:
    accepted = (
        result.auc_advantage_pp >= result.min_auc_advantage_pp
        and result.n_params >= result.min_params
    )
    verdict = "accepted" if accepted else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 060 SUMMARY",
            f"{'=' * 60}",
            f"Params: {result.n_params:,} (min {result.min_params:,})",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Logistic val AUC: {result.logistic_val_auc:.4f}",
            f"LargeNanoMLP val AUC: {result.nano_val_auc:.4f}",
            f"Advantage: {result.auc_advantage_pp:.2f} pp (gate ≥ {result.min_auc_advantage_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 060 — LargeNanoMLP on ACYD Brazil soybean")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_060(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    passed = (
        result.auc_advantage_pp >= result.min_auc_advantage_pp
        and result.n_params >= result.min_params
    )
    return 0 if passed else 1


def _build_results_md(result: LargeNanoAcydResult) -> str:
    from datetime import date

    verdict = (
        "accepted"
        if result.auc_advantage_pp >= result.min_auc_advantage_pp
        else "rejected"
    )
    return "\n".join(
        [
            "# Results — EXP 060: LargeNanoMLP on ACYD Brazil soybean",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate",
            "",
            f"- Params: **{result.n_params:,}**",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Logistic val AUC: **{result.logistic_val_auc:.4f}**",
            f"- LargeNanoMLP val AUC: **{result.nano_val_auc:.4f}**",
            f"- Advantage: **{result.auc_advantage_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — val AUC beats logistic by ≥ {result.min_auc_advantage_pp} pp.",
            "",
            "## Limitations",
            "- ACYD temporal val (2019–2021); test years ≥2022 untouched.",
            "- Agro-climate benchmark — not operational planting advice.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
