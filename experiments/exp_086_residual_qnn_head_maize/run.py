"""
EXP 086 — Residual-skip QNN vs plain QNN on frozen distill ResidualNano (ACYD maize).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_086_residual_qnn_head_maize/run.py \\
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
from src.quantum.residual_nano_hybrid import ResidualNanoHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_086_residual_qnn_head_maize"
EXP_ID = "exp_086"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResidualQnnHeadResult:
    n_trainable_plain: int
    n_trainable_residual: int
    n_train_rows: int
    n_val_rows: int
    classical_val_auc: float
    plain_qnn_val_auc: float
    residual_qnn_val_auc: float
    residual_vs_plain_pp: float
    plain_vs_classical_pp: float
    residual_vs_classical_pp: float
    min_residual_vs_plain_pp: float
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


def _build_hybrid(input_dim: int, cfg: dict, *, residual_skip: bool) -> ResidualNanoHybrid:
    return ResidualNanoHybrid(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 2)),
        reupload=bool(cfg.get("reupload", True)),
        residual_skip=residual_skip,
        backbone_device="cuda" if _training_uses_cuda() else "cpu",
    )


def _train_hybrid(
    *,
    model: ResidualNanoHybrid,
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
    auc = float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])
    return auc, n_train


def _gate_passed(result: ResidualQnnHeadResult) -> bool:
    primary = result.residual_vs_plain_pp >= result.min_residual_vs_plain_pp
    parity_plain = result.plain_vs_classical_pp >= result.min_vs_classical_pp
    parity_res = result.residual_vs_classical_pp >= result.min_vs_classical_pp
    return primary and parity_plain and parity_res


def run_exp_086(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ResidualQnnHeadResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    min_res_pp = float(cfg.get("min_residual_vs_plain_pp", 0.5))
    min_vs_cl = float(cfg.get("min_vs_classical_pp", -1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 086 — Residual QNN skip vs plain | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'}"
        )
        print(
            f"Gate: residual ≥ plain + {min_res_pp} pp | "
            f"both ≥ classical − {abs(min_vs_cl)} pp"
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

    plain = _build_hybrid(input_dim, cfg, residual_skip=False)
    plain_auc, n_plain = _train_hybrid(
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
    if verbose:
        print(f"Plain QNN AUC={plain_auc:.4f} | trainable={n_plain:,}", flush=True)

    residual = _build_hybrid(input_dim, cfg, residual_skip=True)
    residual_auc, n_res = _train_hybrid(
        model=residual,
        state_dict=state_dict,
        x_train=x_train_t,
        y_train=y_train_t,
        x_val=x_val_t,
        y_val=y_val_t,
        cfg=cfg,
        profile=profile,
        seed=seed,
        model_name="residual_qnn_head",
    )
    res_vs_plain = (residual_auc - plain_auc) * 100.0
    plain_vs_cl = (plain_auc - classical_auc) * 100.0
    res_vs_cl = (residual_auc - classical_auc) * 100.0
    elapsed = time.perf_counter() - t0

    if verbose:
        status = "OK" if res_vs_plain >= min_res_pp else "FAIL"
        print(
            f"Residual QNN AUC={residual_auc:.4f} | Δ vs plain={res_vs_plain:.2f} pp [{status}] | "
            f"trainable={n_res:,}",
            flush=True,
        )

    log_event(
        "info",
        "exp_086 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        classical_val_auc=round(classical_auc, 6),
        plain_qnn_val_auc=round(plain_auc, 6),
        residual_qnn_val_auc=round(residual_auc, 6),
        residual_vs_plain_pp=round(res_vs_plain, 3),
        plain_vs_classical_pp=round(plain_vs_cl, 3),
        residual_vs_classical_pp=round(res_vs_cl, 3),
        elapsed_s=round(elapsed, 3),
    )

    return ResidualQnnHeadResult(
        n_trainable_plain=n_plain,
        n_trainable_residual=n_res,
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        classical_val_auc=classical_auc,
        plain_qnn_val_auc=plain_auc,
        residual_qnn_val_auc=residual_auc,
        residual_vs_plain_pp=res_vs_plain,
        plain_vs_classical_pp=plain_vs_cl,
        residual_vs_classical_pp=res_vs_cl,
        min_residual_vs_plain_pp=min_res_pp,
        min_vs_classical_pp=min_vs_cl,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: ResidualQnnHeadResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 086 SUMMARY",
            f"{'=' * 60}",
            f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
            f"Classical AUC: {result.classical_val_auc:.4f}",
            f"Plain QNN AUC: {result.plain_qnn_val_auc:.4f} "
            f"(Δ vs classical {result.plain_vs_classical_pp:.2f} pp)",
            f"Residual QNN AUC: {result.residual_qnn_val_auc:.4f} "
            f"(Δ vs classical {result.residual_vs_classical_pp:.2f} pp)",
            f"Residual vs plain: {result.residual_vs_plain_pp:.2f} pp "
            f"(gate ≥ {result.min_residual_vs_plain_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: ResidualQnnHeadResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    return "\n".join(
        [
            "# Results — EXP 086: Residual-skip QNN vs plain QNN (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Classical distill ResidualNano AUC: **{result.classical_val_auc:.4f}**",
            f"- Plain QNN AUC: **{result.plain_qnn_val_auc:.4f}** "
            f"(Δ classical {result.plain_vs_classical_pp:.2f} pp | "
            f"trainable {result.n_trainable_plain:,})",
            f"- Residual-skip QNN AUC: **{result.residual_qnn_val_auc:.4f}** "
            f"(Δ classical {result.residual_vs_classical_pp:.2f} pp | "
            f"trainable {result.n_trainable_residual:,})",
            f"- Residual vs plain: **{result.residual_vs_plain_pp:.2f} pp** "
            f"(gate ≥ {result.min_residual_vs_plain_pp})",
            f"- Parity floor: ≥ classical − {abs(result.min_vs_classical_pp)} pp",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — Phase B H-Q2.1 residual QNN skip.",
            "",
            "## Limitations",
            "- Frozen distill backbone; PennyLane TorchLayer on CPU.",
            "- Hybrid fine-tune row budget may be capped for QNN wall-time on 4060.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 086 — residual QNN skip on maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_086(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
