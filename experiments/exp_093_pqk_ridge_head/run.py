"""
EXP 093 — Projected quantum kernel ridge head vs logistic (ACYD maize H-Q2.6).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_093_pqk_ridge_head/run.py \\
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_parquet import load_open_parquet_splits
from src.quantum.projected_quantum_kernel import (
    ProjectedQuantumFeatureEncoder,
    fit_kernel_ridge_scores,
    fit_nystroem_logistic_proba,
)
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_093_pqk_ridge_head"
EXP_ID = "exp_093"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PqkRidgeResult:
    n_train_rows: int
    n_val_rows: int
    n_projection_features: int
    logistic_val_auc: float
    projection_logistic_val_auc: float
    kernel_ridge_val_auc: float
    nystroem_logistic_val_auc: float
    histgb_val_auc: float
    ridge_vs_logistic_pp: float
    min_vs_logistic_pp: float
    feature_extract_s: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _gate_passed(result: PqkRidgeResult) -> bool:
    return result.ridge_vs_logistic_pp >= result.min_vs_logistic_pp


def run_exp_093(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> PqkRidgeResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 1))
    ridge_alpha = float(cfg.get("ridge_alpha", 1.0))
    ridge_gamma = cfg.get("ridge_gamma")
    gamma = float(ridge_gamma) if ridge_gamma is not None else None
    nystroem_components = int(cfg.get("nystroem_components", 256))
    min_vs_logistic = float(cfg.get("min_vs_logistic_pp", 0.5))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    progress_every = int(cfg.get("feature_progress_every", 0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 093 — PQK KernelRidge vs logistic | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: KernelRidge ≥ logistic + {min_vs_logistic} pp")
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

    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    logistic_auc = float(roc_auc_score(y_val, logistic.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"Logistic (raw) AUC={logistic_auc:.4f}", flush=True)

    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"HistGB AUC={histgb_auc:.4f} (honesty floor)", flush=True)

    encoder = ProjectedQuantumFeatureEncoder(
        input_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        seed=seed,
    )
    t_feat = time.perf_counter()
    if verbose:
        print(
            f"Extracting {encoder.n_features}-d 1-local PQK projections "
            f"(qubits={n_qubits}, layers={n_layers})…",
            flush=True,
        )
    phi_train = encoder.transform(x_train, progress_every=progress_every)
    phi_val = encoder.transform(x_val, progress_every=progress_every)
    feature_extract_s = time.perf_counter() - t_feat
    if verbose:
        print(f"Feature extract done in {feature_extract_s:.1f}s", flush=True)

    proj_logistic = LogisticRegression(max_iter=500, random_state=seed)
    proj_logistic.fit(phi_train, y_train)
    proj_logistic_auc = float(
        roc_auc_score(y_val, proj_logistic.predict_proba(phi_val)[:, 1])
    )
    if verbose:
        print(f"Logistic (φ projections) AUC={proj_logistic_auc:.4f}", flush=True)

    ridge_scores = fit_kernel_ridge_scores(
        phi_train,
        y_train,
        phi_val,
        alpha=ridge_alpha,
        gamma=gamma,
    )
    ridge_auc = float(roc_auc_score(y_val, ridge_scores))
    if verbose:
        print(f"KernelRidge (PQK/RBF) AUC={ridge_auc:.4f}", flush=True)

    nys_proba = fit_nystroem_logistic_proba(
        phi_train,
        y_train,
        phi_val,
        n_components=nystroem_components,
        gamma=gamma,
        seed=seed,
    )
    nys_auc = float(roc_auc_score(y_val, nys_proba))
    if verbose:
        print(f"Nyström→logistic (PQK linear head) AUC={nys_auc:.4f}", flush=True)

    ridge_vs_log = (ridge_auc - logistic_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if ridge_vs_log >= min_vs_logistic else "FAIL"
        print(
            f"Δ KernelRidge vs logistic={ridge_vs_log:.2f} pp [{status}]",
            flush=True,
        )

    log_event(
        "info",
        "exp_093 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        logistic_val_auc=round(logistic_auc, 6),
        projection_logistic_val_auc=round(proj_logistic_auc, 6),
        kernel_ridge_val_auc=round(ridge_auc, 6),
        nystroem_logistic_val_auc=round(nys_auc, 6),
        histgb_val_auc=round(histgb_auc, 6),
        ridge_vs_logistic_pp=round(ridge_vs_log, 3),
        feature_extract_s=round(feature_extract_s, 3),
        elapsed_s=round(elapsed, 3),
    )

    return PqkRidgeResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_projection_features=int(encoder.n_features),
        logistic_val_auc=logistic_auc,
        projection_logistic_val_auc=proj_logistic_auc,
        kernel_ridge_val_auc=ridge_auc,
        nystroem_logistic_val_auc=nys_auc,
        histgb_val_auc=histgb_auc,
        ridge_vs_logistic_pp=ridge_vs_log,
        min_vs_logistic_pp=min_vs_logistic,
        feature_extract_s=round(feature_extract_s, 3),
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: PqkRidgeResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 093 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Logistic (raw) AUC: {result.logistic_val_auc:.4f}",
            f"Logistic (φ) AUC: {result.projection_logistic_val_auc:.4f}",
            f"KernelRidge PQK AUC: {result.kernel_ridge_val_auc:.4f} "
            f"(Δ logistic {result.ridge_vs_logistic_pp:.2f} pp)",
            f"Nyström→logistic AUC: {result.nystroem_logistic_val_auc:.4f}",
            f"HistGB AUC: {result.histgb_val_auc:.4f}",
            f"Gate: ≥ logistic + {result.min_vs_logistic_pp} pp",
            f"Feature extract: {result.feature_extract_s}s | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: PqkRidgeResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 093: Projected quantum kernel ridge head (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Projection dim: **{result.n_projection_features}** (1-local X/Y/Z on 4q)",
            f"- LogisticRegression (raw) AUC: **{result.logistic_val_auc:.4f}**",
            f"- LogisticRegression (φ projections) AUC: **{result.projection_logistic_val_auc:.4f}**",
            f"- KernelRidge RBF on φ AUC: **{result.kernel_ridge_val_auc:.4f}**",
            f"- Nyström→logistic AUC: **{result.nystroem_logistic_val_auc:.4f}**",
            f"- HistGB (honesty) AUC: **{result.histgb_val_auc:.4f}**",
            f"- KernelRidge vs logistic: **{result.ridge_vs_logistic_pp:.2f} pp** "
            f"(gate ≥ +{result.min_vs_logistic_pp})",
            f"- Feature extract: **{result.feature_extract_s}s** | Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase B H-Q2.6 projected quantum kernel ridge head.",
            "",
            "## Limitations",
            "- Analytic default.qubit projections (infinite-shot), not hardware shots.",
            "- Soft PQK (1-local projections + classical RBF), not full fidelity kernel.",
            "- Train rows capped for per-row QNode wall-time on RTX 4060.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 093 — PQK KernelRidge on maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_093(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
