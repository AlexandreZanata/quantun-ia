"""
EXP 070 — LargeNanoMLP on GoBug file-level defects vs logistic baseline (C3 anchor).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_070_large_nano_gobug/run.py --profile publication --write-results
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

EXP_KEY = "exp_070_large_nano_gobug"
EXP_ID = "exp_070"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LargeNanoGobugResult:
    n_params: int
    n_train_rows: int
    n_val_rows: int
    logistic_val_pr_auc: float
    nano_val_pr_auc: float
    pr_advantage_pp: float
    min_pr_advantage_pp: float
    min_params: int
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


def run_exp_070(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> LargeNanoGobugResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "code_defects_gobug_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 27_172)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_822)
    min_pr_pp = float(cfg.get("min_pr_advantage_pp", 2.0))
    min_params = int(cfg.get("min_params", 1_000_000))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 070 — LargeNanoMLP on GoBug | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: val PR-AUC ≥ logistic + {min_pr_pp} pp | params ≥ {min_params:,}")
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
    logistic_pr = _val_pr_auc(logistic, x_val_t, y_val_t)

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
        "exp_070 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        dataset_id=dataset_id,
        n_params=n_params,
        logistic_val_pr_auc=logistic_pr,
        nano_val_pr_auc=nano_pr,
        pr_advantage_pp=round(advantage_pp, 3),
    )

    if verbose:
        status = "OK" if advantage_pp >= min_pr_pp else "FAIL"
        print(
            f"logistic PR-AUC={logistic_pr:.4f} | nano PR-AUC={nano_pr:.4f} | "
            f"Δ={advantage_pp:.2f} pp [{status}] | params={n_params:,} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return LargeNanoGobugResult(
        n_params=n_params,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        logistic_val_pr_auc=logistic_pr,
        nano_val_pr_auc=nano_pr,
        pr_advantage_pp=advantage_pp,
        min_pr_advantage_pp=min_pr_pp,
        min_params=min_params,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: LargeNanoGobugResult) -> bool:
    return (
        result.pr_advantage_pp >= result.min_pr_advantage_pp
        and result.n_params >= result.min_params
    )


def _summarize(result: LargeNanoGobugResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 070 SUMMARY",
            f"{'=' * 60}",
            f"Params: {result.n_params:,} (min {result.min_params:,})",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Logistic val PR-AUC: {result.logistic_val_pr_auc:.4f}",
            f"LargeNanoMLP val PR-AUC: {result.nano_val_pr_auc:.4f}",
            f"Advantage: {result.pr_advantage_pp:.2f} pp (gate ≥ {result.min_pr_advantage_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: LargeNanoGobugResult) -> str:
    verdict = "accepted" if gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 070: LargeNanoMLP on GoBug file-level defects (C3 anchor)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (PR-AUC primary)",
            "",
            f"- Params: **{result.n_params:,}**",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Logistic val PR-AUC: **{result.logistic_val_pr_auc:.4f}**",
            f"- LargeNanoMLP val PR-AUC: **{result.nano_val_pr_auc:.4f}**",
            f"- Advantage: **{result.pr_advantage_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — val PR-AUC beats logistic by ≥ {result.min_pr_advantage_pp} pp.",
            "",
            "## Limitations",
            "- GoBug combined subset (~39K rows); temporal proxy via sha ordering.",
            "- Software defect benchmark — not production static analysis.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 070 — LargeNanoMLP on GoBug (C3 anchor)")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_070(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
