"""
EXP 089 — Measurement-dropout QNN vs plain QNN calibration (ACYD maize H-Q2.4).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_089_measurement_dropout_cal/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.balanced_metrics import expected_calibration_error
from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.measurement_dropout_hybrid import MeasurementDropoutHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_089_measurement_dropout_cal"
EXP_ID = "exp_089"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MeasurementDropoutResult:
    n_train_rows: int
    n_val_rows: int
    n_trainable_plain: int
    n_trainable_dropout: int
    classical_val_auc: float
    plain_qnn_val_auc: float
    plain_qnn_val_ece: float
    dropout_qnn_val_auc: float
    dropout_qnn_val_ece: float
    ece_relative_improvement: float
    auc_delta_pp: float
    min_ece_relative_improvement: float
    min_auc_delta_pp: float
    mc_samples: int
    measurement_dropout: float
    elapsed_s: float
    profile: str


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _load_checkpoint(cfg: dict, root: Path) -> dict[str, torch.Tensor]:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_092"))
    model_name = str(cfg.get("checkpoint_model_name", "residual_nano_distill"))
    seed = int(cfg.get("seed", 42))
    path = root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Distill backbone missing at {path} — run make exp-092-publication first"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_hybrid(
    input_dim: int,
    cfg: dict,
    *,
    measurement_dropout: float,
) -> MeasurementDropoutHybrid:
    return MeasurementDropoutHybrid(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        measurement_dropout=measurement_dropout,
        backbone_device="cuda" if _training_uses_cuda() else "cpu",
    )


def _train_hybrid(
    *,
    model: MeasurementDropoutHybrid,
    state_dict: dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> int:
    model.load_frozen_backbone_from_residual_nano(state_dict)
    model.freeze_backbone()
    n_train = _count_trainable(model)
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.01)),
        batch_size=int(cfg.get("batch_size", 256)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    return n_train


def _collect_probs(
    model: MeasurementDropoutHybrid,
    x_val: torch.Tensor,
    *,
    mc_samples: int,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if model.measurement_dropout > 0.0 and mc_samples > 1:
            probs = model.forward_mc(x_val, n_samples=mc_samples)
        else:
            probs = model(x_val)
    return probs.detach().cpu().numpy().astype(np.float64)


def _gate_passed(result: MeasurementDropoutResult) -> bool:
    ece_ok = result.ece_relative_improvement >= result.min_ece_relative_improvement
    auc_ok = result.auc_delta_pp >= result.min_auc_delta_pp
    return ece_ok and auc_ok


def run_exp_089(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> MeasurementDropoutResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    drop_p = float(cfg.get("measurement_dropout", 0.2))
    mc_samples = int(cfg.get("mc_samples", 16))
    min_ece_rel = float(cfg.get("min_ece_relative_improvement", 0.20))
    min_auc_pp = float(cfg.get("min_auc_delta_pp", -0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 089 — Measurement-dropout QNN | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'} | p={drop_p}"
        )
        print(
            f"Gate: ECE relative ≥ {min_ece_rel:.0%} | "
            f"AUC Δ ≥ {min_auc_pp} pp"
        )
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
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_val_np = y_val.astype(np.float64)

    state_dict = _load_checkpoint(cfg, ROOT)
    classical = ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    classical.load_state_dict(state_dict)
    classical.eval()
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    classical = classical.to(device)
    classical_auc = float(
        evaluate_with_auc(classical, x_val_t.to(device), y_val_t.to(device))["roc_auc"]
    )
    if verbose:
        print(f"Classical distill ResidualNano AUC={classical_auc:.4f}", flush=True)

    plain = _build_hybrid(input_dim, cfg, measurement_dropout=0.0)
    n_plain = _train_hybrid(
        model=plain,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="plain_qnn_head",
    )
    plain_probs = _collect_probs(plain, x_val_t, mc_samples=1)
    plain_auc = float(roc_auc_score(y_val_np, plain_probs))
    plain_ece = float(expected_calibration_error(y_val_np, plain_probs))
    if verbose:
        print(
            f"Plain QNN AUC={plain_auc:.4f} | ECE={plain_ece:.4f} | trainable={n_plain:,}",
            flush=True,
        )

    dropout = _build_hybrid(input_dim, cfg, measurement_dropout=drop_p)
    n_drop = _train_hybrid(
        model=dropout,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="measurement_dropout_qnn",
    )
    drop_probs = _collect_probs(dropout, x_val_t, mc_samples=mc_samples)
    drop_auc = float(roc_auc_score(y_val_np, drop_probs))
    drop_ece = float(expected_calibration_error(y_val_np, drop_probs))
    if plain_ece <= 1e-12:
        ece_rel = 0.0 if drop_ece <= 1e-12 else -1.0
    else:
        ece_rel = (plain_ece - drop_ece) / plain_ece
    auc_pp = (drop_auc - plain_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        ece_status = "OK" if ece_rel >= min_ece_rel else "FAIL"
        auc_status = "OK" if auc_pp >= min_auc_pp else "FAIL"
        print(
            f"Dropout QNN AUC={drop_auc:.4f} | ECE={drop_ece:.4f} | "
            f"ECE rel={ece_rel:.1%} [{ece_status}] | "
            f"ΔAUC={auc_pp:.2f} pp [{auc_status}] | trainable={n_drop:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_089 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        classical_val_auc=round(classical_auc, 6),
        plain_qnn_val_auc=round(plain_auc, 6),
        plain_qnn_val_ece=round(plain_ece, 6),
        dropout_qnn_val_auc=round(drop_auc, 6),
        dropout_qnn_val_ece=round(drop_ece, 6),
        ece_relative_improvement=round(ece_rel, 6),
        auc_delta_pp=round(auc_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    return MeasurementDropoutResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_trainable_plain=n_plain,
        n_trainable_dropout=n_drop,
        classical_val_auc=classical_auc,
        plain_qnn_val_auc=plain_auc,
        plain_qnn_val_ece=plain_ece,
        dropout_qnn_val_auc=drop_auc,
        dropout_qnn_val_ece=drop_ece,
        ece_relative_improvement=ece_rel,
        auc_delta_pp=auc_pp,
        min_ece_relative_improvement=min_ece_rel,
        min_auc_delta_pp=min_auc_pp,
        mc_samples=mc_samples,
        measurement_dropout=drop_p,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: MeasurementDropoutResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 089 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Classical AUC: {result.classical_val_auc:.4f}",
            f"Plain QNN: AUC={result.plain_qnn_val_auc:.4f} | ECE={result.plain_qnn_val_ece:.4f}",
            f"Dropout QNN (p={result.measurement_dropout}, MC={result.mc_samples}): "
            f"AUC={result.dropout_qnn_val_auc:.4f} | ECE={result.dropout_qnn_val_ece:.4f}",
            f"ECE relative improvement: {result.ece_relative_improvement:.1%} "
            f"(gate ≥ {result.min_ece_relative_improvement:.0%})",
            f"AUC Δ: {result.auc_delta_pp:.2f} pp (gate ≥ {result.min_auc_delta_pp})",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: MeasurementDropoutResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 089: Measurement-dropout QNN calibration (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Classical distill ResidualNano AUC: **{result.classical_val_auc:.4f}**",
            f"- Plain QNN: AUC **{result.plain_qnn_val_auc:.4f}** | "
            f"ECE **{result.plain_qnn_val_ece:.4f}** | trainable {result.n_trainable_plain:,}",
            f"- Measurement-dropout QNN (p={result.measurement_dropout}, "
            f"MC={result.mc_samples}): AUC **{result.dropout_qnn_val_auc:.4f}** | "
            f"ECE **{result.dropout_qnn_val_ece:.4f}** | trainable {result.n_trainable_dropout:,}",
            f"- ECE relative improvement: **{result.ece_relative_improvement:.1%}** "
            f"(gate ≥ {result.min_ece_relative_improvement:.0%})",
            f"- AUC Δ (dropout − plain): **{result.auc_delta_pp:.2f} pp** "
            f"(gate ≥ {result.min_auc_delta_pp})",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase B H-Q2.4 measurement-dropout calibration.",
            "",
            "## Limitations",
            "- Frozen distill backbone; PennyLane TorchLayer on CPU.",
            "- Hybrid fine-tune row budget capped for QNN wall-time on 4060.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 089 — measurement-dropout QNN")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_089(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
