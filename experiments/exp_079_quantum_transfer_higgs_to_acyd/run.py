"""
EXP 079 — Cross-domain quantum head transfer (HIGGS C1 → ACYD C4).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_079_quantum_transfer_higgs_to_acyd/run.py --profile publication --write-results
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

from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_079_quantum_transfer_higgs_to_acyd"
EXP_ID = "exp_079"
ROOT = Path(__file__).resolve().parents[2]
HEAD_KEYS = ("head_proj.", "qlayer.", "post.")


@dataclass(frozen=True)
class TransferAcydResult:
    n_train_rows: int
    n_val_rows: int
    n_trainable_params: int
    scratch_val_auc: float
    transfer_val_auc: float
    transfer_advantage_pp: float
    max_transfer_advantage_pp: float
    source_checkpoint: str
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint missing at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _backbone_path(cfg: dict, root: Path) -> Path:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_060"))
    model_name = str(cfg.get("checkpoint_model_name", "large_nano_mlp"))
    seed = int(cfg.get("seed", 42))
    return root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"


def _source_head_path(cfg: dict, root: Path) -> Path:
    exp_id = str(cfg.get("source_exp_id", "exp_037"))
    model_name = str(cfg.get("source_model_name", "large_nano_hybrid"))
    seed = int(cfg.get("source_seed", cfg.get("seed", 42)))
    return root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"


def _extract_head_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    head = {
        k: v.detach().cpu().clone()
        for k, v in state.items()
        if k.startswith(HEAD_KEYS)
    }
    required = {"head_proj.weight", "head_proj.bias", "qlayer.weights", "post.0.weight", "post.0.bias"}
    missing = required - set(head)
    if missing:
        raise KeyError(f"source hybrid missing head keys: {sorted(missing)}")
    return head


def _build_hybrid(
    input_dim: int,
    cfg: dict,
    backbone_state: dict[str, torch.Tensor],
) -> LargeNanoHybrid:
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
    model.load_frozen_backbone_from_large_nano(backbone_state)
    model.freeze_backbone()
    model.backbone.to(model._backbone_device)
    return model


def _train_head(
    model: LargeNanoHybrid,
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    seed: int,
    profile: str,
    model_name: str,
) -> float:
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.01)),
        batch_size=int(cfg.get("batch_size", 512)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    return float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])


def gate_passed(result: TransferAcydResult) -> bool:
    """Hypothesis confirmed when transfer does not beat scratch by ≥ max_transfer_advantage_pp."""
    return result.transfer_advantage_pp < result.max_transfer_advantage_pp


def run_exp_079(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> TransferAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    max_adv = float(cfg.get("max_transfer_advantage_pp", 0.5))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 079 — Quantum head transfer HIGGS→ACYD | profile={profile} | "
            f"train={n_train or 'all'}"
        )
        print(f"Hypothesis gate: transfer − scratch < +{max_adv} pp (honest negative)")
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

    backbone_path = _backbone_path(cfg, ROOT)
    source_path = _source_head_path(cfg, ROOT)
    backbone_state = _load_state(backbone_path)
    source_head = _extract_head_state(_load_state(source_path))

    if verbose:
        print(f"C4 backbone: {backbone_path}", flush=True)
        print(f"HIGGS head source: {source_path}", flush=True)
        print("Arm A: scratch head...", flush=True)

    scratch = _build_hybrid(input_dim, cfg, copy.deepcopy(backbone_state))
    n_trainable = sum(p.numel() for p in scratch.parameters() if p.requires_grad)
    scratch_auc = _train_head(
        scratch,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        seed=seed,
        profile=profile,
        model_name="hybrid_scratch_acyd",
    )

    if verbose:
        print(f"Scratch val ROC-AUC={scratch_auc:.4f}", flush=True)
        print("Arm B: transfer head from HIGGS + fine-tune...", flush=True)

    transfer = _build_hybrid(input_dim, cfg, copy.deepcopy(backbone_state))
    missing, unexpected = transfer.load_state_dict(source_head, strict=False)
    unexpected_non_bb = [k for k in unexpected if not k.startswith("backbone.")]
    if unexpected_non_bb:
        raise RuntimeError(f"unexpected head keys from HIGGS hybrid: {unexpected_non_bb}")
    _ = missing
    transfer_auc = _train_head(
        transfer,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        seed=seed + 1,
        profile=profile,
        model_name="hybrid_transfer_higgs_acyd",
    )

    advantage_pp = (transfer_auc - scratch_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_079 quantum transfer summary",
        exp_id=EXP_ID,
        profile=profile,
        scratch_val_auc=round(scratch_auc, 4),
        transfer_val_auc=round(transfer_auc, 4),
        transfer_advantage_pp=round(advantage_pp, 3),
        source_checkpoint=str(source_path),
        elapsed_s=round(elapsed, 3),
    )

    result = TransferAcydResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_trainable_params=n_trainable,
        scratch_val_auc=scratch_auc,
        transfer_val_auc=transfer_auc,
        transfer_advantage_pp=advantage_pp,
        max_transfer_advantage_pp=max_adv,
        source_checkpoint=str(source_path),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "CONFIRMED" if gate_passed(result) else "REJECTED"
        print(
            f"scratch={scratch_auc:.4f} | transfer={transfer_auc:.4f} | "
            f"Δ={advantage_pp:+.2f} pp [{status}] | {elapsed:.1f}s",
            flush=True,
        )

    return result


def _summarize(result: TransferAcydResult) -> str:
    verdict = "honest_negative_confirmed" if gate_passed(result) else "transfer_win_unexpected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 079 SUMMARY",
            f"{'=' * 60}",
            f"Train: {result.n_train_rows:,} | Val: {result.n_val_rows:,}",
            f"Scratch head AUC: {result.scratch_val_auc:.4f}",
            f"Transfer head AUC: {result.transfer_val_auc:.4f}",
            f"Transfer advantage: {result.transfer_advantage_pp:+.2f} pp "
            f"(hypothesis < +{result.max_transfer_advantage_pp} pp)",
            f"Source: {result.source_checkpoint}",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: TransferAcydResult) -> str:
    verdict = (
        "honest negative confirmed"
        if gate_passed(result)
        else "unexpected transfer win (hypothesis rejected)"
    )
    return "\n".join(
        [
            "# Results — EXP 079: Quantum head transfer HIGGS → ACYD (H-Q13)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate (ROC-AUC)",
            "",
            f"- Train rows: **{result.n_train_rows:,}** · Val rows: **{result.n_val_rows:,}**",
            f"- Trainable head params: **{result.n_trainable_params:,}**",
            f"- Scratch head (frozen C4): **{result.scratch_val_auc:.4f}**",
            f"- Transfer head (HIGGS init → fine-tune): **{result.transfer_val_auc:.4f}**",
            f"- Transfer advantage: **{result.transfer_advantage_pp:+.2f} pp**",
            f"- Hypothesis: advantage **< +{result.max_transfer_advantage_pp} pp**",
            f"- Source checkpoint: `{result.source_checkpoint}`",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — cross-domain QNN head transfer on frozen C4.",
            "",
            "## Limitations",
            "- Head-only transfer; C1 backbone shapes incompatible with ACYD input_dim.",
            "- PennyLane QNN on CPU; frozen C4 backbone on CUDA.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 079 — quantum head transfer HIGGS→ACYD (H-Q13)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_079(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
