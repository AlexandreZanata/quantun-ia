"""
EXP 096 — GoBug streaming ResidualNano vs joint train (Phase C / C-T6).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_096_gobug_streaming_nano/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.balanced_metrics import pr_auc
from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import train_model_batched
from src.training.config import load_experiment_config
from src.training.streaming_batches import (
    chronological_batches,
    predict_proba,
    train_streaming_batches,
)
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_096_gobug_streaming_nano"
EXP_ID = "exp_096"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GobugStreamingResult:
    n_train_rows: int
    n_val_rows: int
    n_stream_batches: int
    n_params: int
    logistic_val_pr_auc: float
    joint_val_pr_auc: float
    streaming_val_pr_auc: float
    streaming_vs_joint_pp: float
    min_vs_joint_pp: float
    joint_epochs: int
    epochs_per_batch: int
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _gate_passed(result: GobugStreamingResult) -> bool:
    return result.streaming_vs_joint_pp >= result.min_vs_joint_pp


def _pr_auc_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    score = pr_auc(y_true, probs)
    return float(score) if score is not None else 0.5


def _build_model(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def run_exp_096(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> GobugStreamingResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "code_defects_gobug_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    n_batches = int(cfg.get("n_stream_batches", 8))
    epochs_per_batch = int(cfg.get("epochs_per_batch", 2))
    min_vs_joint = float(cfg.get("min_vs_joint_pp", -1.0))
    lr = float(cfg.get("learning_rate", 0.001))
    batch_size = int(cfg.get("batch_size", 2048))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 096 — GoBug streaming vs joint | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: streaming ≥ joint − {abs(min_vs_joint)} pp PR-AUC")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    # Full load preserves sha-sorted parquet order; prefix cap keeps chronology.
    x_full, y_full, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=None,
        n_val_rows=n_val,
        random_state=seed,
    )
    if n_train is not None and n_train < len(y_full):
        x_train, y_train = x_full[:n_train], y_full[:n_train]
    else:
        x_train, y_train = x_full, y_full

    input_dim = int(x_train.shape[1])
    joint_epochs = int(cfg.get("joint_epochs") or max(n_batches * epochs_per_batch, 1))
    stream_chunks = chronological_batches(x_train, y_train, n_batches=n_batches)
    n_stream = len(stream_chunks)

    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    logistic_pr = _pr_auc_score(y_val, logistic.predict_proba(x_val)[:, 1])
    if verbose:
        print(f"Logistic PR-AUC={logistic_pr:.4f} (honesty)", flush=True)

    joint = _build_model(input_dim, cfg)
    train_model_batched(
        joint,
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        EXP_ID,
        "residual_nano_joint",
        epochs=joint_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        X_val=torch.tensor(x_val, dtype=torch.float32),
        y_val=torch.tensor(y_val, dtype=torch.float32),
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    joint_pr = _pr_auc_score(y_val, predict_proba(joint, x_val))
    if verbose:
        print(f"Joint ResidualNano PR-AUC={joint_pr:.4f}", flush=True)

    stream = _build_model(input_dim, cfg)
    train_streaming_batches(
        stream,
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_streaming",
        n_batches=n_stream,
        epochs_per_batch=epochs_per_batch,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        profile=profile,
    )
    stream_pr = _pr_auc_score(y_val, predict_proba(stream, x_val))
    delta_pp = (stream_pr - joint_pr) * 100.0
    elapsed = time.perf_counter() - t0
    n_params = count_parameters(stream)

    if verbose:
        status = "OK" if delta_pp >= min_vs_joint else "FAIL"
        print(
            f"Streaming PR-AUC={stream_pr:.4f} | Δ vs joint={delta_pp:.2f} pp [{status}] | "
            f"batches={n_stream}",
            flush=True,
        )

    log_event(
        "info",
        "exp_096 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        logistic_val_pr_auc=round(logistic_pr, 6),
        joint_val_pr_auc=round(joint_pr, 6),
        streaming_val_pr_auc=round(stream_pr, 6),
        streaming_vs_joint_pp=round(delta_pp, 3),
        n_stream_batches=n_stream,
        elapsed_s=round(elapsed, 3),
    )

    return GobugStreamingResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_stream_batches=n_stream,
        n_params=n_params,
        logistic_val_pr_auc=logistic_pr,
        joint_val_pr_auc=joint_pr,
        streaming_val_pr_auc=stream_pr,
        streaming_vs_joint_pp=delta_pp,
        min_vs_joint_pp=min_vs_joint,
        joint_epochs=joint_epochs,
        epochs_per_batch=epochs_per_batch,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: GobugStreamingResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 096 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,} | "
            f"Batches: {result.n_stream_batches}",
            f"Logistic PR-AUC: {result.logistic_val_pr_auc:.4f}",
            f"Joint PR-AUC: {result.joint_val_pr_auc:.4f} (epochs={result.joint_epochs})",
            f"Streaming PR-AUC: {result.streaming_val_pr_auc:.4f} "
            f"(Δ {result.streaming_vs_joint_pp:.2f} pp | "
            f"{result.epochs_per_batch} ep/batch)",
            f"Gate: ≥ joint − {abs(result.min_vs_joint_pp)} pp",
            f"Params: {result.n_params:,} | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: GobugStreamingResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 096: GoBug streaming ResidualNano",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Stream batches: **{result.n_stream_batches}** | Params: **{result.n_params:,}**",
            f"- Logistic PR-AUC: **{result.logistic_val_pr_auc:.4f}**",
            f"- Joint ResidualNano PR-AUC: **{result.joint_val_pr_auc:.4f}** "
            f"({result.joint_epochs} epochs)",
            f"- Streaming ResidualNano PR-AUC: **{result.streaming_val_pr_auc:.4f}** "
            f"({result.epochs_per_batch} epochs/batch)",
            f"- Streaming vs joint: **{result.streaming_vs_joint_pp:.2f} pp** "
            f"(gate ≥ {result.min_vs_joint_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase C C-T6 GoBug streaming nano.",
            "",
            "## Limitations",
            "- Sha sort is a temporal proxy, not true commit timestamps.",
            "- Naive chronological fine-tune (no replay/EWC).",
            "- GoBug research benchmark — not production defect triage.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 096 — GoBug streaming nano")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_096(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
