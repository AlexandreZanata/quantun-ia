"""
EXP 097 — SPEI-proxy curriculum vs random staged training (ACYD maize D-T3).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_097_spei_curriculum_agro/run.py \\
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
from src.training.spei_curriculum import (
    sort_by_random_order,
    sort_by_spei_difficulty,
    train_staged_curriculum_batched,
)
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_097_spei_curriculum_agro"
EXP_ID = "exp_097"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SpeiCurriculumResult:
    n_train_rows: int
    n_val_rows: int
    n_params: int
    random_val_auc: float
    spei_val_auc: float
    histgb_val_auc: float
    spei_vs_random_pp: float
    min_vs_random_pp: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _gate_passed(result: SpeiCurriculumResult) -> bool:
    return result.spei_vs_random_pp >= result.min_vs_random_pp


def _build_model(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def run_exp_097(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> SpeiCurriculumResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    n_stages = int(cfg.get("curriculum_stages", 4))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 2))
    refine_epochs = int(cfg.get("refine_epochs", 4))
    min_vs_random = float(cfg.get("min_vs_random_pp", 0.5))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 097 — SPEI curriculum vs random | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: SPEI ≥ random + {min_vs_random} pp")
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

    x_rand, y_rand = sort_by_random_order(x_train, y_train, seed=seed)
    model_rand = _build_model(input_dim, cfg)
    rand_out = train_staged_curriculum_batched(
        model_rand,
        x_rand,
        y_rand,
        x_val,
        y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_random_curriculum",
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        refine_epochs=refine_epochs,
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        seed=seed,
        profile=profile,
    )
    random_auc = float(rand_out["val_roc_auc"])
    if verbose:
        print(f"Random curriculum AUC={random_auc:.4f}", flush=True)

    x_spei, y_spei = sort_by_spei_difficulty(x_train, y_train)
    model_spei = _build_model(input_dim, cfg)
    spei_out = train_staged_curriculum_batched(
        model_spei,
        x_spei,
        y_spei,
        x_val,
        y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_spei_curriculum",
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        refine_epochs=refine_epochs,
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        seed=seed,
        profile=profile,
    )
    spei_auc = float(spei_out["val_roc_auc"])
    delta_pp = (spei_auc - random_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if delta_pp >= min_vs_random else "FAIL"
        print(
            f"SPEI curriculum AUC={spei_auc:.4f} | Δ vs random={delta_pp:.2f} pp [{status}]",
            flush=True,
        )

    log_event(
        "info",
        "exp_097 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        random_val_auc=round(random_auc, 6),
        spei_val_auc=round(spei_auc, 6),
        histgb_val_auc=round(histgb_auc, 6),
        spei_vs_random_pp=round(delta_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    return SpeiCurriculumResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_params=int(spei_out["n_params"]),
        random_val_auc=random_auc,
        spei_val_auc=spei_auc,
        histgb_val_auc=histgb_auc,
        spei_vs_random_pp=delta_pp,
        min_vs_random_pp=min_vs_random,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: SpeiCurriculumResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 097 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Random curriculum AUC: {result.random_val_auc:.4f}",
            f"SPEI curriculum AUC: {result.spei_val_auc:.4f} "
            f"(Δ {result.spei_vs_random_pp:.2f} pp)",
            f"HistGB AUC: {result.histgb_val_auc:.4f}",
            f"Gate: ≥ random + {result.min_vs_random_pp} pp",
            f"Params: {result.n_params:,} | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: SpeiCurriculumResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 097: SPEI-proxy curriculum (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- ResidualNano params: **{result.n_params:,}**",
            f"- Random staged curriculum AUC: **{result.random_val_auc:.4f}**",
            f"- SPEI easy→hard curriculum AUC: **{result.spei_val_auc:.4f}**",
            f"- HistGB (honesty) AUC: **{result.histgb_val_auc:.4f}**",
            f"- SPEI vs random: **{result.spei_vs_random_pp:.2f} pp** "
            f"(gate ≥ +{result.min_vs_random_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase D D-T3 SPEI-proxy curriculum.",
            "",
            "## Limitations",
            "- SPEI is precipitation-mean order proxy (feature index 9), not full SPEI.",
            "- Matched staged+refine epoch budget vs random permutation curriculum.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 097 — SPEI curriculum on maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_097(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
