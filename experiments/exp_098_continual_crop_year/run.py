"""
EXP 098 — Continual crop-year fine-tune vs joint ResidualNano (ACYD maize D-T4).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_098_continual_crop_year/run.py \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.continual_crop_year import load_continual_crop_year_splits
from src.training.config import load_experiment_config
from src.training.continual_year import sklearn_auc, train_continual_by_year, train_joint
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_098_continual_crop_year"
EXP_ID = "exp_098"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContinualCropYearResult:
    n_train_rows: int
    n_val_rows: int
    n_train_years: int
    n_params: int
    joint_val_auc: float
    continual_val_auc: float
    backward_mean_auc: float
    histgb_val_auc: float
    continual_vs_joint_pp: float
    min_vs_joint_pp: float
    joint_epochs: int
    epochs_per_year: int
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None) -> int | None:
    if value is None:
        return None
    iv = int(value)
    return None if iv <= 0 else iv


def _gate_passed(result: ContinualCropYearResult) -> bool:
    return result.continual_vs_joint_pp >= result.min_vs_joint_pp


def _build_model(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def run_exp_098(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ContinualCropYearResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"))
    n_val = _resolve_row_cap(cfg.get("n_val_rows"))
    epochs_per_year = int(cfg.get("epochs_per_year", 2))
    min_vs_joint = float(cfg.get("min_vs_joint_pp", -1.0))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    max_chunks = cfg.get("max_feature_chunks")
    max_chunks_i = None if max_chunks is None or int(max_chunks) <= 0 else int(max_chunks)
    lr = float(cfg.get("learning_rate", 0.001))
    batch_size = int(cfg.get("batch_size", 2048))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 098 — Continual crop-year vs joint | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: continual ≥ joint − {abs(min_vs_joint)} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    splits = load_continual_crop_year_splits(
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
        max_feature_chunks=max_chunks_i,
    )
    input_dim = int(splits.x_train.shape[1])
    n_years = len(splits.train_years)
    joint_epochs = int(cfg.get("joint_epochs") or max(n_years * epochs_per_year, 1))

    if verbose:
        print(
            f"Train years={n_years} ({splits.train_years[0]}…{splits.train_years[-1]}) | "
            f"rows={splits.n_train:,} | joint_epochs={joint_epochs}",
            flush=True,
        )

    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(splits.x_train, splits.y_train)
    histgb_auc = sklearn_auc(histgb, splits.x_val, splits.y_val)
    if verbose:
        print(f"HistGB AUC={histgb_auc:.4f} (honesty floor)", flush=True)

    joint_model = _build_model(input_dim, cfg)
    joint_auc = train_joint(
        joint_model,
        splits.x_train,
        splits.y_train,
        splits.x_val,
        splits.y_val,
        exp_id=EXP_ID,
        model_name="residual_nano_joint",
        epochs=joint_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        profile=profile,
    )
    if verbose:
        print(f"Joint ResidualNano AUC={joint_auc:.4f}", flush=True)

    cont_model = _build_model(input_dim, cfg)
    cont_auc, backward_auc = train_continual_by_year(
        cont_model,
        splits.x_train,
        splits.y_train,
        splits.years_train,
        splits.x_val,
        splits.y_val,
        splits.train_years,
        exp_id=EXP_ID,
        model_name="residual_nano_continual",
        epochs_per_year=epochs_per_year,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        seed=seed,
        profile=profile,
    )
    delta_pp = (cont_auc - joint_auc) * 100.0
    elapsed = time.perf_counter() - t0
    n_params = count_parameters(cont_model)

    if verbose:
        status = "OK" if delta_pp >= min_vs_joint else "FAIL"
        print(
            f"Continual AUC={cont_auc:.4f} | Δ vs joint={delta_pp:.2f} pp [{status}] | "
            f"backward_mean={backward_auc:.4f}",
            flush=True,
        )

    log_event(
        "info",
        "exp_098 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        joint_val_auc=round(joint_auc, 6),
        continual_val_auc=round(cont_auc, 6),
        backward_mean_auc=round(float(backward_auc), 6),
        histgb_val_auc=round(histgb_auc, 6),
        continual_vs_joint_pp=round(delta_pp, 3),
        n_train_years=n_years,
        elapsed_s=round(elapsed, 3),
    )

    return ContinualCropYearResult(
        n_train_rows=splits.n_train,
        n_val_rows=splits.n_val,
        n_train_years=n_years,
        n_params=n_params,
        joint_val_auc=joint_auc,
        continual_val_auc=cont_auc,
        backward_mean_auc=float(backward_auc),
        histgb_val_auc=histgb_auc,
        continual_vs_joint_pp=delta_pp,
        min_vs_joint_pp=min_vs_joint,
        joint_epochs=joint_epochs,
        epochs_per_year=epochs_per_year,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: ContinualCropYearResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 098 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,} | "
            f"Years: {result.n_train_years}",
            f"Joint AUC: {result.joint_val_auc:.4f} (epochs={result.joint_epochs})",
            f"Continual AUC: {result.continual_val_auc:.4f} "
            f"(Δ {result.continual_vs_joint_pp:.2f} pp | "
            f"{result.epochs_per_year} ep/year)",
            f"Backward mean AUC: {result.backward_mean_auc:.4f}",
            f"HistGB AUC: {result.histgb_val_auc:.4f}",
            f"Gate: ≥ joint − {abs(result.min_vs_joint_pp)} pp",
            f"Params: {result.n_params:,} | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: ContinualCropYearResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 098: Continual crop-year fine-tune (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Train years: **{result.n_train_years}** | Params: **{result.n_params:,}**",
            f"- Joint ResidualNano AUC: **{result.joint_val_auc:.4f}** "
            f"({result.joint_epochs} epochs)",
            f"- Continual year-by-year AUC: **{result.continual_val_auc:.4f}** "
            f"({result.epochs_per_year} epochs/year)",
            f"- Backward mean AUC (prior years): **{result.backward_mean_auc:.4f}**",
            f"- HistGB (honesty) AUC: **{result.histgb_val_auc:.4f}**",
            f"- Continual vs joint: **{result.continual_vs_joint_pp:.2f} pp** "
            f"(gate ≥ {result.min_vs_joint_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase D D-T4 continual crop-year fine-tune.",
            "",
            "## Limitations",
            "- Naive fine-tune without EWC/replay (honest lower bound).",
            "- Year column rebuilt from raw ACYD into processed/continual_v1.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 098 — continual crop-year maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_098(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
