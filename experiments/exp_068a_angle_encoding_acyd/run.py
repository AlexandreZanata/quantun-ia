"""
EXP 068a — Seasonal angle vs amplitude QNN head on frozen ACYD C4.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_068a_angle_encoding_acyd/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid_acyd import LargeNanoHybridAcyd
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_068a_angle_encoding_acyd"
EXP_ID = "exp_068a"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AngleEncodingAcydResult:
    n_backbone_params: int
    n_angle_params: int
    n_amplitude_params: int
    n_train_rows: int
    n_val_rows: int
    classical_val_auc: float
    angle_val_auc: float
    amplitude_val_auc: float
    angle_vs_classical_pp: float
    angle_vs_amplitude_pp: float
    min_vs_classical_pp: float
    min_vs_amplitude_pp: float
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


def _eval_classical_head(
    state_dict: dict[str, torch.Tensor],
    *,
    input_dim: int,
    cfg: dict,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    classical = LargeNanoMLP(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
    )
    classical.load_state_dict(state_dict)
    classical.eval()
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    classical = classical.to(device)
    metrics = evaluate_with_auc(classical, x_val.to(device), y_val.to(device))
    return float(metrics["roc_auc"])


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_hybrid(cfg: dict, *, input_dim: int, encoding: str) -> LargeNanoHybridAcyd:
    model = LargeNanoHybridAcyd(
        input_dim=input_dim,
        hidden1=int(cfg.get("hidden1", 2048)),
        hidden2=int(cfg.get("hidden2", 512)),
        hidden3=int(cfg.get("hidden3", 64)),
        dropout=float(cfg.get("dropout", 0.3)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        encoding=encoding,  # type: ignore[arg-type]
        backbone_device="cuda" if _training_uses_cuda() else "cpu",
    )
    return model


def _train_seasonal_head(
    model: LargeNanoHybridAcyd,
    state_dict: dict[str, torch.Tensor],
    *,
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    model_name: str,
    profile: str,
    seed: int,
) -> None:
    model.load_frozen_backbone_from_large_nano(state_dict)
    model.freeze_backbone()
    model.backbone.to(model._backbone_device)
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


def run_exp_068a(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> AngleEncodingAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    min_vs_classical_pp = float(cfg.get("min_vs_classical_pp", 0.5))
    min_vs_amplitude_pp = float(cfg.get("min_vs_amplitude_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 068a — Seasonal angle vs amplitude QNN | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: angle ≥ classical + {min_vs_classical_pp} pp")
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

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    state_dict = _load_classical_checkpoint(cfg, ROOT)
    classical_auc = _eval_classical_head(
        state_dict,
        input_dim=input_dim,
        cfg=cfg,
        x_val=x_val_t,
        y_val=y_val_t,
    )

    angle_model = _build_hybrid(cfg, input_dim=input_dim, encoding="angle_seasonal")
    n_backbone = count_parameters(angle_model.backbone)
    if verbose:
        print("Training seasonal angle head...", flush=True)
    _train_seasonal_head(
        angle_model,
        state_dict,
        cfg=cfg,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        model_name="large_nano_hybrid_acyd_angle",
        profile=profile,
        seed=seed,
    )
    n_angle = _count_trainable(angle_model)
    angle_metrics = evaluate_with_auc(angle_model, x_val_t, y_val_t)
    angle_auc = float(angle_metrics["roc_auc"])

    amplitude_model = _build_hybrid(cfg, input_dim=input_dim, encoding="amplitude")
    amplitude_model.load_frozen_backbone_from_large_nano(state_dict)
    amplitude_model.freeze_backbone()
    if verbose:
        print("Training seasonal amplitude head...", flush=True)
    _train_seasonal_head(
        amplitude_model,
        state_dict,
        cfg=cfg,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        model_name="large_nano_hybrid_acyd_amplitude",
        profile=profile,
        seed=seed,
    )
    n_amplitude = _count_trainable(amplitude_model)
    amplitude_metrics = evaluate_with_auc(amplitude_model, x_val_t, y_val_t)
    amplitude_auc = float(amplitude_metrics["roc_auc"])

    angle_vs_classical_pp = (angle_auc - classical_auc) * 100.0
    angle_vs_amplitude_pp = (angle_auc - amplitude_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_068a seasonal encoding summary",
        exp_id=EXP_ID,
        profile=profile,
        classical_val_auc=classical_auc,
        angle_val_auc=angle_auc,
        amplitude_val_auc=amplitude_auc,
        angle_vs_classical_pp=round(angle_vs_classical_pp, 3),
        angle_vs_amplitude_pp=round(angle_vs_amplitude_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        print(
            f"classical AUC={classical_auc:.4f} | angle AUC={angle_auc:.4f} | "
            f"amplitude AUC={amplitude_auc:.4f}",
            flush=True,
        )
        print(
            f"angle vs classical: {angle_vs_classical_pp:.2f} pp | "
            f"angle vs amplitude: {angle_vs_amplitude_pp:.2f} pp | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return AngleEncodingAcydResult(
        n_backbone_params=n_backbone,
        n_angle_params=n_angle,
        n_amplitude_params=n_amplitude,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        classical_val_auc=classical_auc,
        angle_val_auc=angle_auc,
        amplitude_val_auc=amplitude_auc,
        angle_vs_classical_pp=angle_vs_classical_pp,
        angle_vs_amplitude_pp=angle_vs_amplitude_pp,
        min_vs_classical_pp=min_vs_classical_pp,
        min_vs_amplitude_pp=min_vs_amplitude_pp,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: AngleEncodingAcydResult) -> bool:
    return (
        result.angle_vs_classical_pp >= result.min_vs_classical_pp
        and result.angle_vs_amplitude_pp >= result.min_vs_amplitude_pp
    )


def _summarize(result: AngleEncodingAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 068a SUMMARY",
            f"{'=' * 60}",
            f"Frozen backbone params: {result.n_backbone_params:,}",
            f"Angle head trainable params: {result.n_angle_params:,}",
            f"Amplitude head trainable params: {result.n_amplitude_params:,}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Classical head val ROC-AUC: {result.classical_val_auc:.4f}",
            f"Angle seasonal val ROC-AUC: {result.angle_val_auc:.4f}",
            f"Amplitude seasonal val ROC-AUC: {result.amplitude_val_auc:.4f}",
            f"Angle vs classical: {result.angle_vs_classical_pp:.2f} pp "
            f"(gate ≥ {result.min_vs_classical_pp} pp)",
            f"Angle vs amplitude: {result.angle_vs_amplitude_pp:.2f} pp "
            f"(gate ≥ {result.min_vs_amplitude_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: AngleEncodingAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 068a: Seasonal Angle Encoding on ACYD (H-Q8)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gates (ROC-AUC)",
            "",
            f"- Classical head val ROC-AUC: **{result.classical_val_auc:.4f}**",
            f"- Angle seasonal val ROC-AUC: **{result.angle_val_auc:.4f}**",
            f"- Amplitude seasonal val ROC-AUC: **{result.amplitude_val_auc:.4f}**",
            f"- Angle vs classical: **{result.angle_vs_classical_pp:.2f} pp**",
            f"- Angle vs amplitude: **{result.angle_vs_amplitude_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — seasonal angle encoding vs classical and amplitude baselines.",
            "",
            "## Limitations",
            "- Cyclic features from scaled in-season weather means (37-dim parquet).",
            "- Single seed publication; QNN sim on CPU.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 068a — Seasonal angle encoding on ACYD")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_068a(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
