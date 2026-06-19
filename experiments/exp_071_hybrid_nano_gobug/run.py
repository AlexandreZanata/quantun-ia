"""
EXP 071 — Frozen LargeNanoMLP (GoBug C3) + 4-qubit hybrid head.

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_071_hybrid_nano_gobug/run.py --profile publication --write-results
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

from src.application.balanced_metrics import pr_auc
from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.batched_trainer import train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters, predict

EXP_KEY = "exp_071_hybrid_nano_gobug"
EXP_ID = "exp_071"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HybridGobugResult:
    n_backbone_params: int
    n_trainable_params: int
    n_train_rows: int
    n_val_rows: int
    classical_val_pr_auc: float
    hybrid_val_pr_auc: float
    vs_classical_pp: float
    min_vs_classical_pp: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _load_classical_checkpoint(cfg: dict, root: Path) -> dict[str, torch.Tensor]:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_070"))
    model_name = str(cfg.get("checkpoint_model_name", "large_nano_mlp"))
    seed = int(cfg.get("seed", 42))
    weights_path = root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"GoBug backbone checkpoint missing at {weights_path} — run make exp-070-publication first"
        )
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _val_pr_auc(model: torch.nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        probs = predict(model, x_val.to(device)).detach().cpu().numpy()
    labels = y_val.detach().cpu().numpy()
    score = pr_auc(labels, probs)
    return float(score) if score is not None else 0.5


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
    return _val_pr_auc(classical, x_val, y_val)


def _count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_exp_071(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> HybridGobugResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "code_defects_gobug_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 27_172)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_822)
    min_vs_classical_pp = float(cfg.get("min_vs_classical_pp", -1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 071 — Hybrid QNN head on frozen GoBug backbone | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: hybrid val PR-AUC ≥ classical − {abs(min_vs_classical_pp)} pp")
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
    classical_pr = _eval_classical_head(
        state_dict,
        input_dim=input_dim,
        cfg=cfg,
        x_val=x_val_t,
        y_val=y_val_t,
    )

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
    n_backbone = count_parameters(model.backbone)
    model.load_frozen_backbone_from_large_nano(state_dict)
    model.freeze_backbone()
    model.backbone.to(model._backbone_device)

    n_trainable = _count_trainable(model)
    if verbose:
        print(
            f"Backbone params (frozen): {n_backbone:,} | Trainable head: {n_trainable:,}",
            flush=True,
        )

    train_model_batched(
        model,
        x_train_t,
        y_train_t,
        EXP_ID,
        "large_nano_hybrid_gobug",
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

    hybrid_pr = _val_pr_auc(model, x_val_t, y_val_t)
    vs_classical_pp = (hybrid_pr - classical_pr) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_071 hybrid head summary",
        exp_id=EXP_ID,
        profile=profile,
        n_backbone_params=n_backbone,
        n_trainable_params=n_trainable,
        classical_val_pr_auc=classical_pr,
        hybrid_val_pr_auc=hybrid_pr,
        vs_classical_pp=round(vs_classical_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if vs_classical_pp >= min_vs_classical_pp else "FAIL"
        print(
            f"classical PR-AUC={classical_pr:.4f} | hybrid PR-AUC={hybrid_pr:.4f} | "
            f"Δ={vs_classical_pp:.2f} pp [{status}] | elapsed={elapsed:.1f}s",
            flush=True,
        )

    return HybridGobugResult(
        n_backbone_params=n_backbone,
        n_trainable_params=n_trainable,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        classical_val_pr_auc=classical_pr,
        hybrid_val_pr_auc=hybrid_pr,
        vs_classical_pp=vs_classical_pp,
        min_vs_classical_pp=min_vs_classical_pp,
        elapsed_s=round(elapsed, 3),
    )


def gate_passed(result: HybridGobugResult) -> bool:
    return result.vs_classical_pp >= result.min_vs_classical_pp


def _summarize(result: HybridGobugResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 071 SUMMARY",
            f"{'=' * 60}",
            f"Frozen backbone params: {result.n_backbone_params:,}",
            f"Trainable head params: {result.n_trainable_params:,}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Classical head PR-AUC: {result.classical_val_pr_auc:.4f}",
            f"Hybrid head PR-AUC: {result.hybrid_val_pr_auc:.4f}",
            f"Δ vs classical: {result.vs_classical_pp:.2f} pp "
            f"(gate ≥ {result.min_vs_classical_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: HybridGobugResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 071: Hybrid QNN Head on Frozen GoBug LargeNanoMLP (C3)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (PR-AUC)",
            "",
            f"- Frozen backbone params: **{result.n_backbone_params:,}**",
            f"- Trainable head params: **{result.n_trainable_params:,}**",
            f"- Train rows: **{result.n_train_rows:,}**",
            f"- Val rows: **{result.n_val_rows:,}**",
            f"- Classical head PR-AUC: **{result.classical_val_pr_auc:.4f}**",
            f"- Hybrid head PR-AUC: **{result.hybrid_val_pr_auc:.4f}**",
            f"- Δ vs classical: **{result.vs_classical_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — hybrid val PR-AUC vs frozen classical head "
            f"(gate ≥ {result.min_vs_classical_pp} pp).",
            "",
            "## Limitations",
            "- Temporal val split only (sha-order proxy); test split untouched.",
            "- QNN sim on CPU; classical backbone on CUDA.",
            "- Software defect benchmark — not production static analysis.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 071 — Hybrid QNN head on frozen GoBug LargeNanoMLP (C3)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_071(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
