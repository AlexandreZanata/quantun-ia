"""
EXP 075 — GV-ALR vs fixed LR on frozen hybrid QNN head (NIHR C2 replication).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_075_adaptive_hybrid_nihr/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.application.balanced_metrics import pr_auc
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.adaptive_lr import AdaptiveLRConfig
from src.training.batched_trainer import train_model_batched, train_model_batched_adaptive
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import predict

EXP_KEY = "exp_075_adaptive_hybrid_nihr"
EXP_ID = "exp_075"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AdaptiveHybridNihrResult:
    fixed_epochs: int
    adaptive_epochs: int
    max_epoch_fraction: float
    max_pr_delta_pp: float
    fixed_val_pr_auc: float
    adaptive_val_pr_auc: float
    pr_delta_pp: float
    fixed_wall_s: float
    adaptive_wall_s: float
    wall_time_ratio: float
    n_train_rows: int
    n_val_rows: int
    n_trainable_params: int


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _load_backbone_checkpoint(cfg: dict, root: Path) -> dict[str, torch.Tensor]:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_069"))
    model_name = str(cfg.get("checkpoint_model_name", "large_nano_mlp"))
    seed = int(cfg.get("seed", 42))
    weights_path = root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"backbone checkpoint missing at {weights_path} — run make exp-069-publication first"
        )
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _build_hybrid(input_dim: int, cfg: dict, state_dict: dict[str, torch.Tensor]) -> LargeNanoHybrid:
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
    model.load_frozen_backbone_from_large_nano(state_dict)
    model.freeze_backbone()
    model.backbone.to(model._backbone_device)
    return model


def _adaptive_config(cfg: dict) -> AdaptiveLRConfig:
    adapt = dict(cfg.get("adaptive_lr", {}))
    return AdaptiveLRConfig(
        base_lr=float(adapt.get("base_lr", cfg.get("learning_rate", 0.01))),
        var_target=float(adapt.get("var_target", cfg.get("adaptive_var_target", 0.015))),
        min_scale=float(adapt.get("min_scale", 0.25)),
        max_scale=float(adapt.get("max_scale", 4.0)),
        warmup_epochs=int(adapt.get("warmup_epochs", 1)),
        adapt_every=int(adapt.get("adapt_every", 1)),
    )


def _val_pr_auc(model: torch.nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = predict(model, x_val.to(device)).detach().cpu().numpy()
    labels = y_val.detach().cpu().numpy()
    score = pr_auc(labels, probs)
    return float(score) if score is not None else 0.5


def gate_passed(result: AdaptiveHybridNihrResult) -> bool:
    epoch_ok = result.adaptive_epochs <= int(result.fixed_epochs * result.max_epoch_fraction)
    pr_ok = abs(result.pr_delta_pp) <= result.max_pr_delta_pp
    return epoch_ok and pr_ok


def run_exp_075(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> AdaptiveHybridNihrResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "nihr_cv_synthetic_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 70_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 15_000)
    fixed_epochs = int(cfg.get("fixed_epochs", 8))
    adaptive_epochs = int(cfg.get("adaptive_epochs", 5))
    max_epoch_fraction = float(cfg.get("max_epoch_fraction", 0.7))
    max_pr_delta_pp = float(cfg.get("max_pr_delta_pp", 0.3))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 075 — GV-ALR hybrid head (NIHR) | profile={profile} | "
            f"fixed={fixed_epochs}ep vs adaptive={adaptive_epochs}ep"
        )
        print(
            f"Gate: |Δ PR-AUC| ≤ {max_pr_delta_pp} pp · "
            f"adaptive ≤ {max_epoch_fraction:.0%} fixed epochs"
        )
        print(f"{'=' * 60}\n")

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

    backbone_state = _load_backbone_checkpoint(cfg, ROOT)
    train_kwargs = dict(
        batch_size=int(cfg.get("batch_size", 256)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )

    fixed_model = _build_hybrid(input_dim, cfg, backbone_state)
    n_trainable = sum(p.numel() for p in fixed_model.parameters() if p.requires_grad)
    if verbose:
        print(f"Trainable head params: {n_trainable:,}", flush=True)

    t_fixed = time.perf_counter()
    train_model_batched(
        fixed_model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "hybrid_head_fixed",
        epochs=fixed_epochs,
        lr=float(cfg.get("learning_rate", 0.01)),
        **train_kwargs,
    )
    fixed_wall = time.perf_counter() - t_fixed
    fixed_pr = _val_pr_auc(fixed_model, x_val_t, y_val_t)

    adaptive_model = _build_hybrid(input_dim, cfg, copy.deepcopy(backbone_state))
    t_adaptive = time.perf_counter()
    train_model_batched_adaptive(
        adaptive_model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "hybrid_head_adaptive",
        epochs=adaptive_epochs,
        adaptive_config=_adaptive_config(cfg),
        **train_kwargs,
    )
    adaptive_wall = time.perf_counter() - t_adaptive
    adaptive_pr = _val_pr_auc(adaptive_model, x_val_t, y_val_t)

    pr_delta_pp = (adaptive_pr - fixed_pr) * 100.0
    wall_ratio = adaptive_wall / fixed_wall if fixed_wall > 0 else 1.0

    log_event(
        "info",
        "exp_075 adaptive hybrid summary",
        exp_id=EXP_ID,
        profile=profile,
        fixed_epochs=fixed_epochs,
        adaptive_epochs=adaptive_epochs,
        fixed_val_pr_auc=fixed_pr,
        adaptive_val_pr_auc=adaptive_pr,
        pr_delta_pp=round(pr_delta_pp, 3),
        fixed_wall_s=round(fixed_wall, 3),
        adaptive_wall_s=round(adaptive_wall, 3),
        wall_time_ratio=round(wall_ratio, 3),
    )

    result = AdaptiveHybridNihrResult(
        fixed_epochs=fixed_epochs,
        adaptive_epochs=adaptive_epochs,
        max_epoch_fraction=max_epoch_fraction,
        max_pr_delta_pp=max_pr_delta_pp,
        fixed_val_pr_auc=fixed_pr,
        adaptive_val_pr_auc=adaptive_pr,
        pr_delta_pp=pr_delta_pp,
        fixed_wall_s=round(fixed_wall, 3),
        adaptive_wall_s=round(adaptive_wall, 3),
        wall_time_ratio=round(wall_ratio, 3),
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_trainable_params=n_trainable,
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"fixed PR-AUC={fixed_pr:.4f} ({fixed_epochs}ep, {fixed_wall:.1f}s) | "
            f"adaptive PR-AUC={adaptive_pr:.4f} ({adaptive_epochs}ep, {adaptive_wall:.1f}s) | "
            f"Δ={pr_delta_pp:.2f} pp [{status}]",
            flush=True,
        )

    return result


def _summarize(result: AdaptiveHybridNihrResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 075 SUMMARY",
            f"{'=' * 60}",
            f"Fixed LR: {result.fixed_epochs} epochs · PR-AUC {result.fixed_val_pr_auc:.4f} · "
            f"{result.fixed_wall_s}s",
            f"GV-ALR:   {result.adaptive_epochs} epochs · PR-AUC {result.adaptive_val_pr_auc:.4f} · "
            f"{result.adaptive_wall_s}s",
            f"Δ PR-AUC: {result.pr_delta_pp:.2f} pp (gate ≤ {result.max_pr_delta_pp} pp)",
            f"Epoch fraction: {result.adaptive_epochs}/{result.fixed_epochs} "
            f"(gate ≤ {result.max_epoch_fraction:.0%})",
            f"Wall-time ratio: {result.wall_time_ratio:.2f}",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: AdaptiveHybridNihrResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 075: GV-ALR on frozen hybrid QNN head (NIHR C2 replication)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (PR-AUC)",
            "",
            f"- Fixed LR: **{result.fixed_epochs}** epochs · val PR-AUC **{result.fixed_val_pr_auc:.4f}** · "
            f"**{result.fixed_wall_s}s**",
            f"- GV-ALR: **{result.adaptive_epochs}** epochs · val PR-AUC **{result.adaptive_val_pr_auc:.4f}** · "
            f"**{result.adaptive_wall_s}s**",
            f"- Δ PR-AUC: **{result.pr_delta_pp:.2f} pp**",
            f"- Epoch fraction: **{result.adaptive_epochs}/{result.fixed_epochs}**",
            f"- Wall-time ratio: **{result.wall_time_ratio:.2f}**",
            "",
            "## Verdict",
            f"**{verdict}** — |Δ PR-AUC| ≤ {result.max_pr_delta_pp} pp and adaptive epochs ≤ "
            f"{result.max_epoch_fraction:.0%} of fixed.",
            "",
            "## Limitations",
            "- PennyLane QNN sim on CPU; frozen C2 backbone from exp_069.",
            "- Val PR-AUC only; efficiency gate mirrors exp_054.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 075 — GV-ALR vs fixed LR on hybrid QNN head (NIHR)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_075(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
