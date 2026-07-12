"""
EXP 094 — Hard temporal drift ResidualNano vs HistGB on ACYD maize (Phase C / C-T4).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_094_hard_temporal_drift/run.py \\
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
from src.data.hard_drift_acyd import load_hard_drift_maize_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_094_hard_temporal_drift"
EXP_ID = "exp_094"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HardTemporalDriftResult:
    n_train: int
    n_val: int
    n_test: int
    n_params: int
    histgb_val_auc: float
    nano_val_auc: float
    nano_vs_histgb_pp: float
    min_vs_histgb_pp: float
    train_max_year: int
    val_years: tuple[int, ...]
    test_min_year: int
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _resolve_row_cap(value: int | None) -> int | None:
    if value is None:
        return None
    iv = int(value)
    return None if iv <= 0 else iv


def _gate_passed(result: HardTemporalDriftResult) -> bool:
    return result.nano_vs_histgb_pp >= result.min_vs_histgb_pp


def run_exp_094(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> HardTemporalDriftResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    min_vs_hgb = float(cfg.get("min_vs_histgb_pp", -1.0))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"))
    n_val = _resolve_row_cap(cfg.get("n_val_rows"))
    max_chunks = cfg.get("max_feature_chunks")
    max_chunks_i = None if max_chunks is None or int(max_chunks) <= 0 else int(max_chunks)
    train_max_year = int(cfg.get("train_max_year", 2016))
    val_years = tuple(int(y) for y in cfg.get("val_years", [2017, 2018]))
    test_min_year = int(cfg.get("test_min_year", 2022))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 094 — Hard temporal drift | profile={profile} | "
            f"train≤{train_max_year} val={list(val_years)} test≥{test_min_year}"
        )
        print(f"Gate: ResidualNano ≥ HistGB − {abs(min_vs_hgb)} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    # Rebuild only when year protocol matches defaults; otherwise force via ensure kwargs
    from src.data.hard_drift_acyd import ensure_hard_drift_maize_processed

    ensure_hard_drift_maize_processed(
        ROOT,
        force=False,
        max_feature_chunks=max_chunks_i,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
    )
    splits = load_hard_drift_maize_splits(
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
        ensure=False,
        max_feature_chunks=max_chunks_i,
    )
    if verbose:
        print(
            f"Rows train={splits.n_train:,} val={splits.n_val:,} test={splits.n_test:,}",
            flush=True,
        )

    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(splits.x_train, splits.y_train)
    histgb_auc = float(roc_auc_score(splits.y_val, histgb.predict_proba(splits.x_val)[:, 1]))
    if verbose:
        print(f"HistGB hard-drift AUC={histgb_auc:.4f}", flush=True)

    model = ResidualNanoMLP(
        int(splits.x_train.shape[1]),
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    n_params = count_parameters(model)
    x_tr = torch.tensor(splits.x_train, dtype=torch.float32)
    y_tr = torch.tensor(splits.y_train, dtype=torch.float32)
    x_va = torch.tensor(splits.x_val, dtype=torch.float32)
    y_va = torch.tensor(splits.y_val, dtype=torch.float32)
    train_model_batched(
        model,
        x_tr,
        y_tr,
        EXP_ID,
        "residual_nano_hard_drift",
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=int(cfg.get("batch_size", 2048)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_va,
        y_val=y_va,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    nano_auc = float(
        evaluate_with_auc(model.to(device), x_va.to(device), y_va.to(device))["roc_auc"]
    )
    delta = (nano_auc - histgb_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if delta >= min_vs_hgb else "FAIL"
        print(
            f"ResidualNano hard-drift AUC={nano_auc:.4f} | "
            f"Δ vs HistGB={delta:.2f} pp [{status}] | params={n_params:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_094 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        histgb_val_auc=round(histgb_auc, 6),
        nano_val_auc=round(nano_auc, 6),
        nano_vs_histgb_pp=round(delta, 3),
        n_train=splits.n_train,
        n_val=splits.n_val,
        elapsed_s=round(elapsed, 3),
    )

    return HardTemporalDriftResult(
        n_train=splits.n_train,
        n_val=splits.n_val,
        n_test=splits.n_test,
        n_params=n_params,
        histgb_val_auc=histgb_auc,
        nano_val_auc=nano_auc,
        nano_vs_histgb_pp=delta,
        min_vs_histgb_pp=min_vs_hgb,
        train_max_year=train_max_year,
        val_years=val_years,
        test_min_year=test_min_year,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: HardTemporalDriftResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 094 Hard temporal drift (ACYD maize)",
            "",
            f"**Profile:** `{result.profile}`  ",
            f"**Verdict:** {verdict}  ",
            f"**Split:** train ≤ {result.train_max_year} · "
            f"val {list(result.val_years)} · test ≥ {result.test_min_year}  ",
            f"**Rows:** train={result.n_train:,} val={result.n_val:,} test={result.n_test:,}  ",
            f"**Elapsed:** {result.elapsed_s:.1f}s",
            "",
            "| Model | Val ROC-AUC | Notes |",
            "|-------|-------------|-------|",
            f"| HistGB | {result.histgb_val_auc:.4f} | hard-drift val |",
            f"| ResidualNanoMLP | {result.nano_val_auc:.4f} | {result.n_params:,} params |",
            "",
            f"- Δ nano − HistGB = **{result.nano_vs_histgb_pp:.2f} pp** "
            f"(need ≥ {result.min_vs_histgb_pp:.1f})",
            "",
            "## Interpretation",
            "",
            (
                "ResidualNano stayed within the hard-drift gate vs HistGB."
                if verdict == "accepted"
                else "Hard temporal drift widens the boosting gap — "
                "do not claim drift-robust nano without crop-year adaptation."
            ),
            "",
            "## Limitations",
            "",
            "- Rebuild from raw ACYD maize (standard parquet has no year).",
            "- Single seed; agro research benchmark.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    result = run_exp_094(
        profile=args.profile,
        verbose=not args.quiet,
        require_cuda=not args.allow_cpu,
    )
    summary = _summarize(result)
    print(summary)
    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(summary, encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
