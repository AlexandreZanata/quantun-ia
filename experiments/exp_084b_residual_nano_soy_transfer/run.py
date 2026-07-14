"""
EXP 084b — ResidualNano (exp_084 best arch) vs HistGB on ACYD soybean (Phase A / A-T4).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_084b_residual_nano_soy_transfer/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_084b_residual_nano_soy_transfer"
EXP_ID = "exp_084b"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResidualNanoSoyTransferResult:
    n_train_rows: int
    n_val_rows: int
    n_params: int
    histgb_val_auc: float
    residual_val_auc: float
    advantage_vs_histgb_pp: float
    min_auc_advantage_pp: float
    tie_tolerance_pp: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _verdict(result: ResidualNanoSoyTransferResult) -> str:
    delta = result.advantage_vs_histgb_pp
    if delta >= result.min_auc_advantage_pp:
        return "accepted"
    if abs(delta) <= result.tie_tolerance_pp:
        return "honest_tie"
    return "rejected"


def run_exp_084b(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ResidualNanoSoyTransferResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    min_auc_pp = float(cfg.get("min_auc_advantage_pp", 0.5))
    tie_tol = float(cfg.get("tie_tolerance_pp", 0.5))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 084b — ResidualNano soy transfer vs HistGB | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: ResidualNano ≥ HistGB + {min_auc_pp} pp (tie ±{tie_tol} pp)")
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

    t_hgb = time.perf_counter()
    histgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=hgb_max_iter,
        random_state=seed,
    )
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    if verbose:
        print(
            f"HistGB AUC={histgb_auc:.4f} | train_s={time.perf_counter() - t_hgb:.1f}s",
            flush=True,
        )

    model = ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    n_params = count_parameters(model)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    t_nano = time.perf_counter()
    train_model_batched(
        model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "residual_nano_mlp",
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
    residual_auc = float(evaluate_with_auc(model, x_val_t, y_val_t)["roc_auc"])
    if verbose:
        print(
            f"ResidualNanoMLP AUC={residual_auc:.4f} | params={n_params:,} | "
            f"train_s={time.perf_counter() - t_nano:.1f}s",
            flush=True,
        )

    advantage_pp = (residual_auc - histgb_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_084b gate summary",
        exp_id=EXP_ID,
        profile=profile,
        dataset_id=dataset_id,
        histgb_val_auc=round(histgb_auc, 6),
        residual_val_auc=round(residual_auc, 6),
        advantage_vs_histgb_pp=round(advantage_pp, 3),
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        elapsed_s=round(elapsed, 3),
    )

    return ResidualNanoSoyTransferResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_params=n_params,
        histgb_val_auc=histgb_auc,
        residual_val_auc=residual_auc,
        advantage_vs_histgb_pp=advantage_pp,
        min_auc_advantage_pp=min_auc_pp,
        tie_tolerance_pp=tie_tol,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: ResidualNanoSoyTransferResult) -> str:
    verdict = _verdict(result)
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 084b SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"HistGB val AUC: {result.histgb_val_auc:.4f}",
            f"ResidualNano val AUC: {result.residual_val_auc:.4f} | params={result.n_params:,}",
            f"Advantage: {result.advantage_vs_histgb_pp:.2f} pp "
            f"(win ≥ {result.min_auc_advantage_pp} pp | tie ±{result.tie_tolerance_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: ResidualNanoSoyTransferResult) -> str:
    from datetime import date

    verdict = _verdict(result)
    return "\n".join(
        [
            "# Results — EXP 084b: ResidualNano soy transfer vs HistGB",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Params: **{result.n_params:,}**",
            f"- HistGB val ROC-AUC: **{result.histgb_val_auc:.4f}**",
            f"- ResidualNanoMLP val ROC-AUC: **{result.residual_val_auc:.4f}**",
            f"- Advantage: **{result.advantage_vs_histgb_pp:.2f} pp** "
            f"(win ≥ {result.min_auc_advantage_pp} · tie ±{result.tie_tolerance_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase A A-T4 ResidualNano soy transfer.",
            "",
            "## Limitations",
            "- Single seed; from-scratch soy train (architecture transfer, not weight transfer).",
            "- Hyperparameters frozen from exp_084 ResidualNano maize recipe.",
            "- Agro-climate benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 084b — ResidualNano soy transfer vs HistGB"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_084b(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _verdict(result) in {"accepted", "honest_tie"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
