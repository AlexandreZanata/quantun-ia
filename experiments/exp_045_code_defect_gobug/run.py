"""
EXP 045 — LargeNanoMLP on GoBug file-level defects vs logistic baseline.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_045_code_defect_gobug/run.py --profile publication --write-results
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

from src.application.balanced_metrics import pr_auc
from src.classical.large_nano_mlp import LargeNanoMLP
from src.classical.logistic_baseline import LogisticBaseline
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters, predict

EXP_KEY = "exp_045_code_defect_gobug"
EXP_ID = "exp_045"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GobugDefectResult:
    n_params: int
    n_train_rows: int
    n_val_rows: int
    logistic_val_pr_auc: float
    nano_val_pr_auc: float
    pr_advantage_pp: float
    min_val_pr_auc: float
    min_pr_advantage_pp: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _val_pr_auc(model: torch.nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    with torch.no_grad():
        probs = predict(model, x_val).detach().cpu().numpy()
    labels = y_val.detach().cpu().numpy()
    score = pr_auc(labels, probs)
    return float(score) if score is not None else 0.5


def run_exp_045(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> GobugDefectResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "code_defects_gobug_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 30_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 8_000)
    min_val_pr = float(cfg.get("min_val_pr_auc", 0.55))
    min_pr_pp = float(cfg.get("min_pr_advantage_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 045 — GoBug defect baseline | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: val PR-AUC ≥ {min_val_pr} | nano − logistic ≥ {min_pr_pp} pp")
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
    logistic_pr = _val_pr_auc(logistic, x_val_t, y_val_t)

    model = LargeNanoMLP(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 512)),
        hidden2=int(cfg.get("hidden2", 128)),
        hidden3=int(cfg.get("hidden3", 32)),
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
        batch_size=int(cfg.get("batch_size", 1024)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
        device="cuda",
    )
    nano_pr = _val_pr_auc(model, x_val_t, y_val_t)
    advantage_pp = (nano_pr - logistic_pr) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_045 complete",
        exp_id=EXP_ID,
        profile=profile,
        n_params=n_params,
        logistic_val_pr_auc=logistic_pr,
        nano_val_pr_auc=nano_pr,
        pr_advantage_pp=round(advantage_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        print(
            f"logistic PR-AUC={logistic_pr:.4f} | nano PR-AUC={nano_pr:.4f} | "
            f"Δ={advantage_pp:.2f} pp | params={n_params:,} | {elapsed:.1f}s",
            flush=True,
        )

    return GobugDefectResult(
        n_params=n_params,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        logistic_val_pr_auc=logistic_pr,
        nano_val_pr_auc=nano_pr,
        pr_advantage_pp=round(advantage_pp, 3),
        min_val_pr_auc=min_val_pr,
        min_pr_advantage_pp=min_pr_pp,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: GobugDefectResult) -> bool:
    return (
        result.nano_val_pr_auc >= result.min_val_pr_auc
        and result.pr_advantage_pp >= result.min_pr_advantage_pp
    )


def _summarize(result: GobugDefectResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 045 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Logistic val PR-AUC: {result.logistic_val_pr_auc:.4f}",
            f"LargeNanoMLP val PR-AUC: {result.nano_val_pr_auc:.4f} ({result.n_params:,} params)",
            f"nano − logistic: {result.pr_advantage_pp:.2f} pp "
            f"(gate ≥ {result.min_pr_advantage_pp} pp)",
            f"PR-AUC gate: {result.nano_val_pr_auc:.4f} (need ≥ {result.min_val_pr_auc})",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: GobugDefectResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected / inconclusive"
    return "\n".join(
        [
            "# Results — EXP 045: GoBug file-level defect baseline",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Holdout metrics (PR-AUC primary)",
            "",
            "| Model | Val PR-AUC |",
            "|-------|------------|",
            f"| Logistic | {result.logistic_val_pr_auc:.4f} |",
            f"| LargeNanoMLP | {result.nano_val_pr_auc:.4f} |",
            "",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Params: **{result.n_params:,}**",
            f"- nano − logistic: **{result.pr_advantage_pp:.2f} pp**",
            f"- Wall time: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}**",
            "",
            "## Limitations",
            "- go-bug-collector combined subset (~39k rows); temporal proxy via sha ordering.",
            "- Hybrid head ablation deferred to exp_050.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 045 — GoBug defect baseline")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_045(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
