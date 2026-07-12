"""
EXP 099 — Masked climate SSL pretrain → fine-tune vs scratch (ACYD maize D-T5).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_099_ssl_climate_pretrain/run.py \\
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
from src.training.config import load_experiment_config
from src.training.ssl_climate import (
    ResidualNanoSSL,
    copy_encoder_to_residual_nano,
    pretrain_masked_climate,
    train_supervised_residual,
)
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_099_ssl_climate_pretrain"
EXP_ID = "exp_099"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SslClimateResult:
    n_train_rows: int
    n_val_rows: int
    n_params: int
    scratch_val_auc: float
    ssl_val_auc: float
    histgb_val_auc: float
    pretrain_mse: float
    ssl_vs_scratch_pp: float
    min_vs_scratch_pp: float
    pretrain_epochs: int
    finetune_epochs: int
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _gate_passed(result: SslClimateResult) -> bool:
    return result.ssl_vs_scratch_pp >= result.min_vs_scratch_pp


def _build_supervised(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def run_exp_099(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> SslClimateResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    pretrain_epochs = int(cfg.get("pretrain_epochs", 8))
    finetune_epochs = int(cfg.get("finetune_epochs", 12))
    mask_ratio = float(cfg.get("mask_ratio", 0.3))
    min_vs_scratch = float(cfg.get("min_vs_scratch_pp", 0.5))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    lr = float(cfg.get("learning_rate", 0.001))
    pretrain_lr = float(cfg.get("pretrain_lr", lr))
    batch_size = int(cfg.get("batch_size", 2048))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 099 — SSL climate vs scratch | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: SSL ≥ scratch + {min_vs_scratch} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    input_dim = int(x_train.shape[1])

    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"HistGB AUC={histgb_auc:.4f} (honesty floor)", flush=True)

    scratch = _build_supervised(input_dim, cfg)
    scratch_auc = train_supervised_residual(
        scratch,
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_scratch",
        epochs=finetune_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        profile=profile,
    )
    if verbose:
        print(f"Scratch ResidualNano AUC={scratch_auc:.4f}", flush=True)

    ssl = ResidualNanoSSL(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    pre = pretrain_masked_climate(
        ssl,
        x_train,
        exp_id=EXP_ID,
        model_name="residual_nano_ssl_pretrain",
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        batch_size=batch_size,
        mask_ratio=mask_ratio,
        seed=seed,
        profile=profile,
    )
    if verbose:
        print(
            f"SSL pretrain MSE={pre['mse']:.4f} | weather_dims={pre['n_weather']}",
            flush=True,
        )

    ssl_sup = _build_supervised(input_dim, cfg)
    n_copied = copy_encoder_to_residual_nano(ssl, ssl_sup)
    ssl_auc = train_supervised_residual(
        ssl_sup,
        x_train,
        y_train,
        x_val,
        y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_ssl_finetune",
        epochs=finetune_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        profile=profile,
    )
    delta_pp = (ssl_auc - scratch_auc) * 100.0
    elapsed = time.perf_counter() - t0
    n_params = count_parameters(ssl_sup)

    if verbose:
        status = "OK" if delta_pp >= min_vs_scratch else "FAIL"
        print(
            f"SSL fine-tune AUC={ssl_auc:.4f} | Δ vs scratch={delta_pp:.2f} pp [{status}] | "
            f"copied={n_copied:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_099 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        scratch_val_auc=round(scratch_auc, 6),
        ssl_val_auc=round(ssl_auc, 6),
        histgb_val_auc=round(histgb_auc, 6),
        pretrain_mse=round(float(pre["mse"]), 6),
        ssl_vs_scratch_pp=round(delta_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    return SslClimateResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_params=n_params,
        scratch_val_auc=scratch_auc,
        ssl_val_auc=ssl_auc,
        histgb_val_auc=histgb_auc,
        pretrain_mse=float(pre["mse"]),
        ssl_vs_scratch_pp=delta_pp,
        min_vs_scratch_pp=min_vs_scratch,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: SslClimateResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 099 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Scratch AUC: {result.scratch_val_auc:.4f} "
            f"({result.finetune_epochs} supervised epochs)",
            f"SSL fine-tune AUC: {result.ssl_val_auc:.4f} "
            f"(pretrain {result.pretrain_epochs} + fine-tune {result.finetune_epochs})",
            f"Δ SSL − scratch: {result.ssl_vs_scratch_pp:.2f} pp",
            f"Pretrain MSE: {result.pretrain_mse:.4f}",
            f"HistGB AUC: {result.histgb_val_auc:.4f}",
            f"Gate: ≥ scratch + {result.min_vs_scratch_pp} pp",
            f"Params: {result.n_params:,} | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: SslClimateResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 099: Masked climate SSL pretrain (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- ResidualNano params: **{result.n_params:,}**",
            f"- Scratch AUC: **{result.scratch_val_auc:.4f}** "
            f"({result.finetune_epochs} supervised epochs)",
            f"- SSL fine-tune AUC: **{result.ssl_val_auc:.4f}** "
            f"(pretrain {result.pretrain_epochs} + fine-tune {result.finetune_epochs})",
            f"- Pretrain MSE: **{result.pretrain_mse:.4f}**",
            f"- HistGB (honesty) AUC: **{result.histgb_val_auc:.4f}**",
            f"- SSL vs scratch: **{result.ssl_vs_scratch_pp:.2f} pp** "
            f"(gate ≥ +{result.min_vs_scratch_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase D D-T5 masked climate SSL pretrain.",
            "",
            "## Limitations",
            "- Pretext masks seasonal weather aggregates (features 9–36), not raw weekly tensors.",
            "- Matched supervised epochs; SSL spends additional pretrain epochs.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 099 — SSL climate pretrain maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_099(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
