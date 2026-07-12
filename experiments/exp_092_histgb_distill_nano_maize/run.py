"""
EXP 092 — HistGB → ResidualNano soft-label distillation on ACYD maize (Phase D / H-N3).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_092_histgb_distill_nano_maize/run.py \\
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
from src.training.distillation import mix_hard_soft_targets, soft_targets_from_proba
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_092_histgb_distill_nano_maize"
EXP_ID = "exp_092"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HistgbDistillMaizeResult:
    n_train_rows: int
    n_val_rows: int
    n_params: int
    histgb_val_auc: float
    hard_nano_val_auc: float
    distill_nano_val_auc: float
    advantage_vs_teacher_pp: float
    advantage_vs_hard_pp: float
    min_teacher_gap_pp: float
    distill_alpha: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _gate_passed(result: HistgbDistillMaizeResult) -> bool:
    # Student must stay within min_teacher_gap_pp below teacher (gap ≥ -threshold)
    return result.advantage_vs_teacher_pp >= -result.min_teacher_gap_pp


def _build_student(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def _train_student(
    *,
    model: ResidualNanoMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> float:
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    return float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])


def run_exp_092(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> HistgbDistillMaizeResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    min_gap = float(cfg.get("min_teacher_gap_pp", 1.0))
    alpha = float(cfg.get("distill_alpha", 1.0))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 092 — HistGB→ResidualNano distill | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'} | α={alpha}"
        )
        print(f"Gate: distill nano ≥ HistGB − {min_gap} pp (hard-label val)")
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

    histgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=hgb_max_iter,
        random_state=seed,
    )
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    soft_train = soft_targets_from_proba(histgb.predict_proba(x_train))
    y_distill = mix_hard_soft_targets(y_train, soft_train, alpha=alpha)

    if verbose:
        print(f"HistGB teacher AUC={histgb_auc:.4f}", flush=True)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_hard_t = torch.tensor(y_train, dtype=torch.float32)
    y_distill_t = torch.tensor(y_distill, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    hard_model = _build_student(input_dim, cfg)
    n_params = count_parameters(hard_model)
    hard_auc = _train_student(
        model=hard_model,
        x_train=x_train_t,
        y_train=y_hard_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="residual_nano_hard",
    )
    if verbose:
        print(f"Hard ResidualNano AUC={hard_auc:.4f} | params={n_params:,}", flush=True)

    distill_model = _build_student(input_dim, cfg)
    distill_auc = _train_student(
        model=distill_model,
        x_train=x_train_t,
        y_train=y_distill_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="residual_nano_distill",
    )
    vs_teacher = (distill_auc - histgb_auc) * 100.0
    vs_hard = (distill_auc - hard_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if vs_teacher >= -min_gap else "FAIL"
        print(
            f"Distill ResidualNano AUC={distill_auc:.4f} | "
            f"Δ vs HistGB={vs_teacher:.2f} pp | Δ vs hard={vs_hard:.2f} pp [{status}]",
            flush=True,
        )

    log_event(
        "info",
        "exp_092 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        dataset_id=dataset_id,
        histgb_val_auc=round(histgb_auc, 6),
        hard_nano_val_auc=round(hard_auc, 6),
        distill_nano_val_auc=round(distill_auc, 6),
        advantage_vs_teacher_pp=round(vs_teacher, 3),
        advantage_vs_hard_pp=round(vs_hard, 3),
        distill_alpha=alpha,
        n_params=n_params,
        elapsed_s=round(elapsed, 3),
    )

    return HistgbDistillMaizeResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_params=n_params,
        histgb_val_auc=histgb_auc,
        hard_nano_val_auc=hard_auc,
        distill_nano_val_auc=distill_auc,
        advantage_vs_teacher_pp=vs_teacher,
        advantage_vs_hard_pp=vs_hard,
        min_teacher_gap_pp=min_gap,
        distill_alpha=alpha,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: HistgbDistillMaizeResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 092 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Params: {result.n_params:,} | α={result.distill_alpha}",
            f"HistGB teacher AUC: {result.histgb_val_auc:.4f}",
            f"Hard ResidualNano AUC: {result.hard_nano_val_auc:.4f}",
            f"Distill ResidualNano AUC: {result.distill_nano_val_auc:.4f}",
            f"Δ vs teacher: {result.advantage_vs_teacher_pp:.2f} pp "
            f"(gate ≥ −{result.min_teacher_gap_pp} pp)",
            f"Δ vs hard: {result.advantage_vs_hard_pp:.2f} pp",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: HistgbDistillMaizeResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 092: HistGB → ResidualNano distillation (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Student params: **{result.n_params:,}** | distill α: **{result.distill_alpha}**",
            f"- HistGB teacher val AUC: **{result.histgb_val_auc:.4f}**",
            f"- Hard ResidualNano val AUC: **{result.hard_nano_val_auc:.4f}**",
            f"- Distill ResidualNano val AUC: **{result.distill_nano_val_auc:.4f}**",
            f"- Δ vs teacher: **{result.advantage_vs_teacher_pp:.2f} pp** "
            f"(gate ≥ −{result.min_teacher_gap_pp} pp)",
            f"- Δ vs hard control: **{result.advantage_vs_hard_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase D H-N3 soft-label distillation vs HistGB teacher.",
            "",
            "## Limitations",
            "- Soft BCE (not temperature KL); single seed; temporal val only.",
            "- Agro-climate benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 092 — HistGB→ResidualNano distillation on ACYD maize"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_092(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
