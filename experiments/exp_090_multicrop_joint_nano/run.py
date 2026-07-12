"""
EXP 090 — Multi-crop joint ResidualNano (soy+maize) vs maize-solo (Phase C).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_090_multicrop_joint_nano/run.py \\
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
from src.data.multicrop_acyd import load_multicrop_acyd_splits, maize_solo_open_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_090_multicrop_joint_nano"
EXP_ID = "exp_090"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MulticropJointNanoResult:
    n_soy_train: int
    n_maize_train: int
    n_val_maize: int
    n_val_soy: int
    n_params_joint: int
    n_params_solo: int
    histgb_maize_val_auc: float
    solo_maize_val_auc: float
    joint_maize_val_auc: float
    joint_soy_val_auc: float
    joint_vs_solo_pp: float
    min_vs_solo_pp: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _gate_passed(result: MulticropJointNanoResult) -> bool:
    return result.joint_vs_solo_pp >= result.min_vs_solo_pp


def _build_model(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def _train_and_eval(
    *,
    model: ResidualNanoMLP,
    x_train,
    y_train,
    x_val,
    y_val,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> tuple[float, int]:
    n_params = count_parameters(model)
    x_tr = torch.tensor(x_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    x_va = torch.tensor(x_val, dtype=torch.float32)
    y_va = torch.tensor(y_val, dtype=torch.float32)
    train_model_batched(
        model,
        x_tr,
        y_tr,
        EXP_ID,
        model_name,
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
    auc = float(evaluate_with_auc(model.to(device), x_va.to(device), y_va.to(device))["roc_auc"])
    return auc, n_params


def run_exp_090(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> MulticropJointNanoResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seed = int(cfg.get("seed", 42))
    min_vs_solo = float(cfg.get("min_vs_solo_pp", -0.5))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    include_crop = bool(cfg.get("include_crop_indicator", True))

    n_tr_mz = cfg.get("n_train_rows_maize")
    n_tr_soy = cfg.get("n_train_rows_soy")
    n_va_mz = cfg.get("n_val_rows_maize")
    n_va_soy = cfg.get("n_val_rows_soy")
    # Normalize 0 → None (full split)
    def _cap(v):
        if v is None:
            return None
        iv = int(v)
        return None if iv <= 0 else iv

    n_tr_mz, n_tr_soy, n_va_mz, n_va_soy = map(_cap, (n_tr_mz, n_tr_soy, n_va_mz, n_va_soy))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 090 — Multi-crop joint ResidualNano | profile={profile} | "
            f"crop_bit={include_crop}"
        )
        print(f"Gate: joint maize ≥ solo maize − {abs(min_vs_solo)} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    splits = load_multicrop_acyd_splits(
        ROOT,
        n_train_rows_maize=n_tr_mz,
        n_train_rows_soy=n_tr_soy,
        n_val_rows_maize=n_va_mz,
        n_val_rows_soy=n_va_soy,
        random_state=seed,
        include_crop_indicator=include_crop,
    )
    if verbose:
        print(
            f"Train soy={splits.n_soy_train:,} maize={splits.n_maize_train:,} | "
            f"val maize={len(splits.y_val_maize):,} soy={len(splits.y_val_soy):,} | "
            f"input_dim={splits.x_train.shape[1]}",
            flush=True,
        )

    # HistGB honesty on maize-only open splits (37-d native)
    x_h_tr, y_h_tr, x_h_va, y_h_va = maize_solo_open_splits(
        ROOT,
        n_train_rows=n_tr_mz,
        n_val_rows=n_va_mz,
        random_state=seed,
    )
    histgb = HistGradientBoostingClassifier(max_iter=hgb_max_iter, random_state=seed)
    histgb.fit(x_h_tr, y_h_tr)
    histgb_auc = float(roc_auc_score(y_h_va, histgb.predict_proba(x_h_va)[:, 1]))
    if verbose:
        print(f"HistGB maize AUC={histgb_auc:.4f} (honesty)", flush=True)

    solo_model = _build_model(int(splits.x_train_maize_solo.shape[1]), cfg)
    solo_auc, n_solo = _train_and_eval(
        model=solo_model,
        x_train=splits.x_train_maize_solo,
        y_train=splits.y_train_maize_solo,
        x_val=splits.x_val_maize,
        y_val=splits.y_val_maize,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="residual_nano_maize_solo",
    )
    if verbose:
        print(f"Maize-solo ResidualNano AUC={solo_auc:.4f} | params={n_solo:,}", flush=True)

    joint_model = _build_model(int(splits.x_train.shape[1]), cfg)
    joint_maize_auc, n_joint = _train_and_eval(
        model=joint_model,
        x_train=splits.x_train,
        y_train=splits.y_train,
        x_val=splits.x_val_maize,
        y_val=splits.y_val_maize,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="residual_nano_multicrop",
    )
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    joint_soy_auc = float(
        evaluate_with_auc(
            joint_model.to(device),
            torch.tensor(splits.x_val_soy, dtype=torch.float32, device=device),
            torch.tensor(splits.y_val_soy, dtype=torch.float32, device=device),
        )["roc_auc"]
    )
    delta = (joint_maize_auc - solo_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if delta >= min_vs_solo else "FAIL"
        print(
            f"Joint maize AUC={joint_maize_auc:.4f} | soy val={joint_soy_auc:.4f} | "
            f"Δ vs solo={delta:.2f} pp [{status}] | params={n_joint:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_090 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        histgb_maize_val_auc=round(histgb_auc, 6),
        solo_maize_val_auc=round(solo_auc, 6),
        joint_maize_val_auc=round(joint_maize_auc, 6),
        joint_soy_val_auc=round(joint_soy_auc, 6),
        joint_vs_solo_pp=round(delta, 3),
        elapsed_s=round(elapsed, 3),
    )

    return MulticropJointNanoResult(
        n_soy_train=splits.n_soy_train,
        n_maize_train=splits.n_maize_train,
        n_val_maize=len(splits.y_val_maize),
        n_val_soy=len(splits.y_val_soy),
        n_params_joint=n_joint,
        n_params_solo=n_solo,
        histgb_maize_val_auc=histgb_auc,
        solo_maize_val_auc=solo_auc,
        joint_maize_val_auc=joint_maize_auc,
        joint_soy_val_auc=joint_soy_auc,
        joint_vs_solo_pp=delta,
        min_vs_solo_pp=min_vs_solo,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: MulticropJointNanoResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 090 Multi-crop joint ResidualNano",
            "",
            f"**Profile:** `{result.profile}`  ",
            f"**Verdict:** {verdict}  ",
            f"**Train:** soy={result.n_soy_train:,} + maize={result.n_maize_train:,}  ",
            f"**Val:** maize={result.n_val_maize:,} · soy={result.n_val_soy:,}  ",
            f"**Elapsed:** {result.elapsed_s:.1f}s",
            "",
            "| Model | Val ROC-AUC | Notes |",
            "|-------|-------------|-------|",
            f"| HistGB maize (honesty) | {result.histgb_maize_val_auc:.4f} | 37-d native |",
            f"| Maize-solo ResidualNano | {result.solo_maize_val_auc:.4f} | "
            f"{result.n_params_solo:,} params |",
            f"| Joint ResidualNano (maize val) | {result.joint_maize_val_auc:.4f} | "
            f"{result.n_params_joint:,} params |",
            f"| Joint ResidualNano (soy val) | {result.joint_soy_val_auc:.4f} | secondary |",
            "",
            f"- Δ joint − solo = **{result.joint_vs_solo_pp:.2f} pp** "
            f"(need ≥ {result.min_vs_solo_pp:.1f})",
            "",
            "## Interpretation",
            "",
            (
                "Shared soy+maize training preserved maize ranking within the Phase C gate."
                if verdict == "accepted"
                else "Joint multi-crop training hurt maize ranking beyond −0.5 pp — "
                "do not claim cross-crop climate transfer without crop-specific heads."
            ),
            "",
            "## Limitations",
            "",
            "- Hard labels; crop indicator after StandardScaler.",
            "- Single seed; existing temporal splits (no year in processed parquet).",
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

    result = run_exp_090(
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
