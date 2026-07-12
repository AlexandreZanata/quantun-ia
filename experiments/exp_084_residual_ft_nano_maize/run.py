"""
EXP 084 — ResidualNano / NarrowDeepNano / FT-lite vs HistGB on ACYD maize (Phase A / H-N1).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_084_residual_ft_nano_maize/run.py \\
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

from src.classical.ft_lite_nano import FTLiteNano
from src.classical.narrow_deep_nano import NarrowDeepNano
from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_084_residual_ft_nano_maize"
EXP_ID = "exp_084"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArchScore:
    model_key: str
    display_name: str
    roc_auc: float
    n_params: int
    train_s: float


@dataclass(frozen=True)
class ResidualFtNanoMaizeResult:
    n_train_rows: int
    n_val_rows: int
    histgb_val_auc: float
    scores: tuple[ArchScore, ...]
    best_model_key: str
    best_nano_auc: float
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


def _verdict(result: ResidualFtNanoMaizeResult) -> str:
    delta = result.advantage_vs_histgb_pp
    if delta >= result.min_auc_advantage_pp:
        return "accepted"
    if abs(delta) <= result.tie_tolerance_pp:
        return "honest_tie"
    return "rejected"


def _build_models(input_dim: int, cfg: dict) -> list[tuple[str, str, torch.nn.Module]]:
    return [
        (
            "residual_nano_mlp",
            "ResidualNanoMLP",
            ResidualNanoMLP(
                input_dim,
                hidden=int(cfg.get("residual_hidden", 512)),
                n_blocks=int(cfg.get("residual_n_blocks", 3)),
                bottleneck=int(cfg.get("residual_bottleneck", 64)),
                dropout=float(cfg.get("dropout", 0.2)),
            ),
        ),
        (
            "narrow_deep_nano",
            "NarrowDeepNano",
            NarrowDeepNano(
                input_dim,
                width=int(cfg.get("narrow_width", 512)),
                depth=int(cfg.get("narrow_depth", 3)),
                bottleneck=int(cfg.get("narrow_bottleneck", 64)),
                dropout=float(cfg.get("dropout", 0.2)),
            ),
        ),
        (
            "ft_lite_nano",
            "FTLiteNano",
            FTLiteNano(
                input_dim,
                d_token=int(cfg.get("ft_d_token", 32)),
                n_heads=int(cfg.get("ft_n_heads", 4)),
                n_layers=int(cfg.get("ft_n_layers", 2)),
                dropout=float(cfg.get("ft_dropout", 0.1)),
            ),
        ),
    ]


def run_exp_084(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ResidualFtNanoMaizeResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
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
            f"EXP 084 — Residual/FT nano vs HistGB | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: best nano ≥ HistGB + {min_auc_pp} pp (tie ±{tie_tol} pp)")
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
    histgb_proba = histgb.predict_proba(x_val)[:, 1]
    histgb_auc = float(roc_auc_score(y_val, histgb_proba))
    if verbose:
        print(
            f"HistGB AUC={histgb_auc:.4f} | train_s={time.perf_counter() - t_hgb:.1f}s",
            flush=True,
        )

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    scores: list[ArchScore] = []
    for model_key, display_name, model in _build_models(input_dim, cfg):
        n_params = count_parameters(model)
        t_arch = time.perf_counter()
        train_model_batched(
            model,
            x_train_t,
            y_train_t,
            EXP_ID,
            model_key,
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
        metrics = evaluate_with_auc(model, x_val_t, y_val_t)
        auc = float(metrics["roc_auc"])
        train_s = time.perf_counter() - t_arch
        scores.append(
            ArchScore(
                model_key=model_key,
                display_name=display_name,
                roc_auc=auc,
                n_params=n_params,
                train_s=round(train_s, 3),
            )
        )
        if verbose:
            print(
                f"{display_name:20s} AUC={auc:.4f} | params={n_params:,} | "
                f"train_s={train_s:.1f}s",
                flush=True,
            )

    best = max(scores, key=lambda s: s.roc_auc)
    advantage_pp = (best.roc_auc - histgb_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_084 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        dataset_id=dataset_id,
        histgb_val_auc=round(histgb_auc, 6),
        best_model_key=best.model_key,
        best_nano_auc=round(best.roc_auc, 6),
        advantage_vs_histgb_pp=round(advantage_pp, 3),
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        elapsed_s=round(elapsed, 3),
    )

    return ResidualFtNanoMaizeResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        histgb_val_auc=histgb_auc,
        scores=tuple(scores),
        best_model_key=best.model_key,
        best_nano_auc=best.roc_auc,
        advantage_vs_histgb_pp=advantage_pp,
        min_auc_advantage_pp=min_auc_pp,
        tie_tolerance_pp=tie_tol,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: ResidualFtNanoMaizeResult) -> str:
    verdict = _verdict(result)
    lines = [
        f"\n{'=' * 60}",
        "EXP 084 SUMMARY",
        f"{'=' * 60}",
        f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
        f"HistGB val AUC: {result.histgb_val_auc:.4f}",
        f"Best nano: {result.best_model_key} AUC={result.best_nano_auc:.4f}",
        f"Advantage: {result.advantage_vs_histgb_pp:.2f} pp "
        f"(win ≥ {result.min_auc_advantage_pp} pp | tie ±{result.tie_tolerance_pp} pp)",
        f"Elapsed: {result.elapsed_s}s",
        f"Verdict: {verdict}",
        f"{'=' * 60}\n",
    ]
    return "\n".join(lines)


def _build_results_md(result: ResidualFtNanoMaizeResult) -> str:
    from datetime import date

    verdict = _verdict(result)
    rows = "\n".join(
        f"| {s.display_name} | {s.roc_auc:.4f} | {s.n_params:,} | {s.train_s:.1f} |"
        for s in sorted(result.scores, key=lambda x: -x.roc_auc)
    )
    return "\n".join(
        [
            "# Results — EXP 084: Residual / FT-lite nano vs HistGB (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- HistGB val ROC-AUC: **{result.histgb_val_auc:.4f}**",
            f"- Best nano: **{result.best_model_key}** AUC **{result.best_nano_auc:.4f}**",
            f"- Advantage: **{result.advantage_vs_histgb_pp:.2f} pp** "
            f"(win ≥ {result.min_auc_advantage_pp} · tie ±{result.tie_tolerance_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "| Model | Val ROC-AUC | Params | Train (s) |",
            "|-------|-------------|--------|-----------|",
            f"| HistGradientBoosting (sklearn) | {result.histgb_val_auc:.4f} | — | — |",
            rows,
            "",
            "## Verdict",
            f"**{verdict}** — Phase A H-N1 vs HistGB on ACYD maize.",
            "",
            "## Limitations",
            "- Single seed; temporal val only (aligned with exp_081/083).",
            "- Agro-climate benchmark — not operational planting advice.",
            "- If rejected/tie: prefer Phase D distillation (exp_092) before quantum B.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 084 — Residual/FT nano vs HistGB on ACYD maize"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_084(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    # Exit 0 for accepted or honest_tie (documented); 1 only for hard reject
    return 0 if _verdict(result) in {"accepted", "honest_tie"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
