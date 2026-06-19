"""
EXP 068b — Compound stress label: frozen C4 hybrid QNN vs logistic (ACYD H-Q12).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_068b_compound_stress_acyd/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_acyd import load_acyd_compound_stress_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_068b_compound_stress_acyd"
EXP_ID = "exp_068b"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CompoundStressAcydResult:
    n_trainable_params: int
    n_train_rows: int
    n_val_rows: int
    train_positive_rate: float
    val_positive_rate: float
    logistic_val_auc: float
    hybrid_val_auc: float
    vs_logistic_pp: float
    min_vs_logistic_pp: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _load_classical_checkpoint(cfg: dict, root: Path) -> dict[str, torch.Tensor]:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_060"))
    model_name = str(cfg.get("checkpoint_model_name", "large_nano_mlp"))
    seed = int(cfg.get("seed", 42))
    weights_path = root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"ACYD backbone checkpoint missing at {weights_path} — run make exp-060-publication first"
        )
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _fit_logistic_auc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
) -> float:
    logistic = LogisticRegression(max_iter=500, random_state=seed)
    logistic.fit(x_train, y_train)
    probs = logistic.predict_proba(x_val)[:, 1]
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_val)) < 2:
        return 0.5
    return float(roc_auc_score(y_val, probs))


def run_exp_068b(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> CompoundStressAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    min_vs_logistic_pp = float(cfg.get("min_vs_logistic_pp", 1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 068b — Compound stress hybrid vs logistic | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: hybrid val ROC-AUC ≥ logistic + {min_vs_logistic_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train, y_train, x_val, y_val, _x_test, _y_test, _scaler = load_acyd_compound_stress_splits(
        ROOT,
        n_train_rows=n_train,
        n_val_rows=n_val,
        random_state=seed,
        heat_threshold_k=float(cfg.get("heat_threshold_k", 308.15)),
        min_hot_weeks=int(cfg.get("min_hot_weeks", 3)),
        drought_precip_z=float(cfg.get("drought_precip_z", -1.0)),
        drought_weekly_mm=float(cfg.get("drought_weekly_mm", 5.0)),
        min_dry_weeks=int(cfg.get("min_dry_weeks", 4)),
    )
    input_dim = int(x_train.shape[1])

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_pos = float(y_train.mean())
    val_pos = float(y_val.mean())
    if verbose:
        print(
            f"Compound label positive rate — train: {train_pos:.4f} | val: {val_pos:.4f}",
            flush=True,
        )

    logistic_auc = _fit_logistic_auc(x_train, y_train, x_val, y_val, seed=seed)
    if verbose:
        print(f"Logistic val ROC-AUC: {logistic_auc:.4f}", flush=True)

    state_dict = _load_classical_checkpoint(cfg, ROOT)
    model = LargeNanoHybrid(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        backbone_device="cuda" if _training_uses_cuda() else "cpu",
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.load_frozen_backbone_from_large_nano(state_dict)
    model.freeze_backbone()
    model.backbone.to(model._backbone_device)

    if verbose:
        print(f"Trainable head params: {n_trainable:,}", flush=True)

    train_model_batched(
        model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "large_nano_hybrid_acyd_compound",
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.01)),
        batch_size=int(cfg.get("batch_size", 256)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )

    hybrid_metrics = evaluate_with_auc(model, x_val_t, y_val_t)
    hybrid_auc = float(hybrid_metrics["roc_auc"])
    vs_logistic_pp = (hybrid_auc - logistic_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_068b compound stress summary",
        exp_id=EXP_ID,
        profile=profile,
        train_positive_rate=round(train_pos, 6),
        val_positive_rate=round(val_pos, 6),
        logistic_val_auc=logistic_auc,
        hybrid_val_auc=hybrid_auc,
        vs_logistic_pp=round(vs_logistic_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if vs_logistic_pp >= min_vs_logistic_pp else "FAIL"
        print(
            f"logistic AUC={logistic_auc:.4f} | hybrid AUC={hybrid_auc:.4f} | "
            f"Δ={vs_logistic_pp:.2f} pp [{status}] | elapsed={elapsed:.1f}s",
            flush=True,
        )

    return CompoundStressAcydResult(
        n_trainable_params=n_trainable,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        train_positive_rate=train_pos,
        val_positive_rate=val_pos,
        logistic_val_auc=logistic_auc,
        hybrid_val_auc=hybrid_auc,
        vs_logistic_pp=vs_logistic_pp,
        min_vs_logistic_pp=min_vs_logistic_pp,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: CompoundStressAcydResult) -> bool:
    return result.vs_logistic_pp >= result.min_vs_logistic_pp


def _summarize(result: CompoundStressAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 068b SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Train positive rate: {result.train_positive_rate:.4f}",
            f"Val positive rate: {result.val_positive_rate:.4f}",
            f"Logistic val ROC-AUC: {result.logistic_val_auc:.4f}",
            f"Hybrid val ROC-AUC: {result.hybrid_val_auc:.4f}",
            f"Δ vs logistic: {result.vs_logistic_pp:.2f} pp "
            f"(gate ≥ {result.min_vs_logistic_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: CompoundStressAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 068b: Compound Stress Label on ACYD (H-Q12)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (ROC-AUC vs logistic)",
            "",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Train positive rate: **{result.train_positive_rate:.4f}**",
            f"- Val positive rate: **{result.val_positive_rate:.4f}**",
            f"- Logistic val ROC-AUC: **{result.logistic_val_auc:.4f}**",
            f"- Hybrid val ROC-AUC: **{result.hybrid_val_auc:.4f}**",
            f"- Δ vs logistic: **{result.vs_logistic_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — compound-stress hybrid vs logistic "
            f"(gate ≥ {result.min_vs_logistic_pp} pp).",
            "",
            "## Limitations",
            "- Drought stress via train-fitted precipitation z-score (SPEI proxy).",
            "- Temporal val only; highly imbalanced compound label.",
            "- QNN sim on CPU; classical backbone on CUDA.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 068b — Compound stress hybrid QNN vs logistic on ACYD"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_068b(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
