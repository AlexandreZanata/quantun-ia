"""
EXP 095 — CY-Bench maize US sample: ResidualNano vs HistGB (Phase C / C-T5).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_095_cybench_maize_slice/run.py \\
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
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_095_cybench_maize_slice"
EXP_ID = "exp_095"
ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "cybench_maize_us_v1"


@dataclass(frozen=True)
class CybenchMaizeResult:
    n_train: int
    n_val: int
    n_features: int
    n_params: int
    histgb_val_auc: float
    nano_val_auc: float
    nano_vs_histgb_pp: float
    min_vs_histgb_pp: float
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


def _gate_passed(result: CybenchMaizeResult) -> bool:
    return result.nano_vs_histgb_pp >= result.min_vs_histgb_pp


def run_exp_095(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> CybenchMaizeResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"))
    n_val = _resolve_row_cap(cfg.get("n_val_rows"))
    min_pp = float(cfg.get("min_vs_histgb_pp", -1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 095 — CY-Bench maize US | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: ResidualNano ≥ HistGB − {abs(min_pp)} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
        DATASET_ID,
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
    )
    n_features = int(x_train.shape[1])

    histgb = HistGradientBoostingClassifier(
        max_iter=int(cfg.get("histgb_max_iter", 100)),
        random_state=seed,
    )
    histgb.fit(x_train, y_train)
    histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))
    if verbose:
        print(f"HistGB AUC={histgb_auc:.4f}", flush=True)

    model = ResidualNanoMLP(
        n_features,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    n_params = count_parameters(model)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    if _training_uses_cuda():
        # Batched trainer places model; keep tensors CPU for trainer API consistency
        pass

    train_model_batched(
        model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "residual_nano_cybench",
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
    nano_auc = float(evaluate_with_auc(model, x_val_t, y_val_t)["roc_auc"])
    delta_pp = (nano_auc - histgb_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if delta_pp >= min_pp else "FAIL"
        print(
            f"ResidualNano AUC={nano_auc:.4f} | Δ vs HistGB={delta_pp:.2f} pp [{status}] | "
            f"params={n_params:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_095 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        histgb_val_auc=round(histgb_auc, 6),
        nano_val_auc=round(nano_auc, 6),
        nano_vs_histgb_pp=round(delta_pp, 3),
        n_features=n_features,
        elapsed_s=round(elapsed, 3),
    )

    return CybenchMaizeResult(
        n_train=len(y_train),
        n_val=len(y_val),
        n_features=n_features,
        n_params=n_params,
        histgb_val_auc=histgb_auc,
        nano_val_auc=nano_auc,
        nano_vs_histgb_pp=delta_pp,
        min_vs_histgb_pp=min_pp,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: CybenchMaizeResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 095 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train:,} | Val rows: {result.n_val:,} | "
            f"Features: {result.n_features}",
            f"HistGB AUC: {result.histgb_val_auc:.4f}",
            f"ResidualNano AUC: {result.nano_val_auc:.4f} "
            f"(Δ {result.nano_vs_histgb_pp:.2f} pp | gate ≥ {result.min_vs_histgb_pp})",
            f"Params: {result.n_params:,} | Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: CybenchMaizeResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 095: CY-Bench maize US slice (ResidualNano vs HistGB)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "**Dataset:** `cybench_maize_us_v1` (AgML sample US designed features, EUPL-1.2)",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train:,}** | Val rows: **{result.n_val:,}** | "
            f"Features: **{result.n_features}**",
            f"- HistGB AUC: **{result.histgb_val_auc:.4f}**",
            f"- ResidualNano AUC: **{result.nano_val_auc:.4f}** "
            f"(params {result.n_params:,})",
            f"- Δ nano − HistGB: **{result.nano_vs_histgb_pp:.2f} pp** "
            f"(gate ≥ {result.min_vs_histgb_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase C C-T5 CY-Bench maize sample slice.",
            "",
            "## Limitations",
            "- AgML sample US slice only (full Zenodo maize archive not downloaded).",
            "- Binary low-yield proxy — not official CY-Bench regression nRMSE/R².",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 095 — CY-Bench maize slice")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_095(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
