"""
EXP 091 — Circuit-cut effective 6q head vs classical head (ACYD maize H-Q2.5).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_091_circuit_cut_6q/run.py \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.circuit_cut_hybrid import ClassicalBottleneckHead, CircuitCutSixQubitHybrid
from src.quantum.residual_nano_hybrid import ResidualNanoHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_091_circuit_cut_6q"
EXP_ID = "exp_091"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CircuitCutResult:
    n_train_rows: int
    n_val_rows: int
    n_trainable_classical: int
    n_trainable_plain4: int
    n_trainable_cut: int
    classical_val_auc: float
    plain4_val_auc: float
    cut_val_auc: float
    cut_vs_classical_pp: float
    cut_vs_plain4_pp: float
    min_vs_classical_pp: float
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


def _train_any(
    *,
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> tuple[float, int]:
    model.load_frozen_backbone_from_residual_nano(state_dict)  # type: ignore[attr-defined]
    model.freeze_backbone()  # type: ignore[attr-defined]
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
    auc = float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])
    return auc, n_train


def _gate_passed(result: CircuitCutResult) -> bool:
    return result.cut_vs_classical_pp >= result.min_vs_classical_pp


def run_exp_091(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> CircuitCutResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    min_vs_cl = float(cfg.get("min_vs_classical_pp", -1.0))
    bb = "cuda" if _training_uses_cuda() else "cpu"

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 091 — Circuit-cut 6q | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(f"Gate: cut ≥ classical − {abs(min_vs_cl)} pp")
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
    state_dict = _load_checkpoint(cfg, ROOT)

    # Full distill AUC (reference only)
    distill = ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    distill.load_state_dict(state_dict)
    distill.eval()
    device = torch.device("cuda" if _training_uses_cuda() else "cpu")
    distill = distill.to(device)
    distill_auc = float(
        evaluate_with_auc(distill, x_val_t.to(device), y_val_t.to(device))["roc_auc"]
    )
    if verbose:
        print(f"Full distill ResidualNano AUC={distill_auc:.4f} (reference)", flush=True)

    classical = ClassicalBottleneckHead(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        backbone_device=bb,
    )
    classical_auc, n_cl = _train_any(
        model=classical,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="classical_bottleneck_head",
    )
    if verbose:
        print(f"Classical head AUC={classical_auc:.4f} | trainable={n_cl:,}", flush=True)

    plain4 = ResidualNanoHybrid(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        n_qubits=4,
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        residual_skip=False,
        backbone_device=bb,
    )
    plain4_auc, n_p4 = _train_any(
        model=plain4,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="plain_4q_head",
    )
    if verbose:
        print(f"Plain 4q AUC={plain4_auc:.4f} | trainable={n_p4:,}", flush=True)

    cut = CircuitCutSixQubitHybrid(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        backbone_device=bb,
    )
    cut_auc, n_cut = _train_any(
        model=cut,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="circuit_cut_6q",
    )
    cut_vs_cl = (cut_auc - classical_auc) * 100.0
    cut_vs_p4 = (cut_auc - plain4_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if cut_vs_cl >= min_vs_cl else "FAIL"
        print(
            f"Circuit-cut 6q AUC={cut_auc:.4f} | Δ vs classical={cut_vs_cl:.2f} pp [{status}] | "
            f"Δ vs plain4={cut_vs_p4:.2f} pp | trainable={n_cut:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_091 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        distill_val_auc=round(distill_auc, 6),
        classical_val_auc=round(classical_auc, 6),
        plain4_val_auc=round(plain4_auc, 6),
        cut_val_auc=round(cut_auc, 6),
        cut_vs_classical_pp=round(cut_vs_cl, 3),
        cut_vs_plain4_pp=round(cut_vs_p4, 3),
        elapsed_s=round(elapsed, 3),
    )

    return CircuitCutResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_trainable_classical=n_cl,
        n_trainable_plain4=n_p4,
        n_trainable_cut=n_cut,
        classical_val_auc=classical_auc,
        plain4_val_auc=plain4_auc,
        cut_val_auc=cut_auc,
        cut_vs_classical_pp=cut_vs_cl,
        cut_vs_plain4_pp=cut_vs_p4,
        min_vs_classical_pp=min_vs_cl,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: CircuitCutResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 091 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Classical head AUC: {result.classical_val_auc:.4f}",
            f"Plain 4q AUC: {result.plain4_val_auc:.4f}",
            f"Circuit-cut 6q AUC: {result.cut_val_auc:.4f} "
            f"(Δ classical {result.cut_vs_classical_pp:.2f} pp | "
            f"Δ plain4 {result.cut_vs_plain4_pp:.2f} pp)",
            f"Gate: ≥ classical − {abs(result.min_vs_classical_pp)} pp",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: CircuitCutResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 091: Circuit-cut effective 6q head (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Classical bottleneck head AUC: **{result.classical_val_auc:.4f}** "
            f"(trainable {result.n_trainable_classical:,})",
            f"- Plain 4q re-upload AUC: **{result.plain4_val_auc:.4f}** "
            f"(trainable {result.n_trainable_plain4:,})",
            f"- Circuit-cut 2×4q (effective 6q) AUC: **{result.cut_val_auc:.4f}** "
            f"(trainable {result.n_trainable_cut:,})",
            f"- Cut vs classical: **{result.cut_vs_classical_pp:.2f} pp** "
            f"(gate ≥ {result.min_vs_classical_pp})",
            f"- Cut vs plain 4q: **{result.cut_vs_plain4_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase B H-Q2.5 circuit-cut 6q effective head.",
            "",
            "## Limitations",
            "- Soft overlapping-fragment cut (not full tomography reconstruction).",
            "- Frozen distill backbone; PennyLane TorchLayer on CPU.",
            "- Hybrid fine-tune row budget capped for QNN wall-time on 4060.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 091 — circuit-cut 6q on maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_091(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
