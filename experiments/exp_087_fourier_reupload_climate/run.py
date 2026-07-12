"""
EXP 087 — Fourier re-upload vs flat angle head on frozen distill ResidualNano (ACYD maize).

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_087_fourier_reupload_climate/run.py \\
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
from src.quantum.fourier_reupload_hybrid import FourierReuploadHybrid
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_087_fourier_reupload_climate"
EXP_ID = "exp_087"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FourierReuploadResult:
    n_train_rows: int
    n_val_rows: int
    classical_val_auc: float
    flat_auc_by_layers: dict[int, float]
    fourier_auc_by_layers: dict[int, float]
    rung_wins: int
    n_rungs: int
    deepest_layers: int
    flat_deep_vs_classical_pp: float
    fourier_deep_vs_classical_pp: float
    min_rung_wins: int
    min_vs_classical_pp: float
    n_trainable_flat: int
    n_trainable_fourier: int
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


def _parse_layers(cfg: dict) -> list[int]:
    raw = cfg.get("reupload_layers", [1, 2, 3])
    layers = [int(x) for x in raw]
    if not layers:
        raise ValueError("reupload_layers must be non-empty")
    return layers


def _build_hybrid(
    input_dim: int,
    cfg: dict,
    *,
    encoding: str,
    n_layers: int,
) -> FourierReuploadHybrid:
    return FourierReuploadHybrid(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=n_layers,
        encoding=encoding,
        n_frequencies=int(cfg.get("n_frequencies", 2)),
        backbone_device="cuda" if _training_uses_cuda() else "cpu",
    )


def _train_hybrid(
    *,
    model: FourierReuploadHybrid,
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


def _gate_passed(result: FourierReuploadResult) -> bool:
    primary = result.rung_wins >= result.min_rung_wins
    parity_flat = result.flat_deep_vs_classical_pp >= result.min_vs_classical_pp
    parity_fourier = result.fourier_deep_vs_classical_pp >= result.min_vs_classical_pp
    return primary and parity_flat and parity_fourier


def run_exp_087(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> FourierReuploadResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    layers = _parse_layers(cfg)
    min_wins = int(cfg.get("min_rung_wins", 2))
    min_vs_cl = float(cfg.get("min_vs_classical_pp", -1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 087 — Fourier vs flat re-upload | profile={profile} | "
            f"train={n_train or 'all'} | val={n_val or 'all'} | layers={layers}"
        )
        print(
            f"Gate: Fourier wins ≥ {min_wins}/{len(layers)} rungs | "
            f"deepest ≥ classical − {abs(min_vs_cl)} pp"
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

    flat_auc: dict[int, float] = {}
    fourier_auc: dict[int, float] = {}
    n_flat = 0
    n_fourier = 0

    for n_layers in layers:
        flat = _build_hybrid(input_dim, cfg, encoding="flat", n_layers=n_layers)
        flat_score, n_flat = _train_hybrid(
            model=flat,
            state_dict=state_dict,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            cfg=cfg,
            profile=profile,
            seed=seed,
            model_name=f"flat_reupload_L{n_layers}",
        )
        flat_auc[n_layers] = flat_score
        if verbose:
            print(
                f"Flat L={n_layers} AUC={flat_score:.4f} | trainable={n_flat:,}",
                flush=True,
            )

        fourier = _build_hybrid(input_dim, cfg, encoding="fourier", n_layers=n_layers)
        fourier_score, n_fourier = _train_hybrid(
            model=fourier,
            state_dict=state_dict,
            x_train=x_train_t,
            y_train=y_train_t,
            x_val=x_val_t,
            y_val=y_val_t,
            cfg=cfg,
            profile=profile,
            seed=seed,
            model_name=f"fourier_reupload_L{n_layers}",
        )
        fourier_auc[n_layers] = fourier_score
        win = fourier_score > flat_score
        if verbose:
            delta_pp = (fourier_score - flat_score) * 100.0
            print(
                f"Fourier L={n_layers} AUC={fourier_score:.4f} | "
                f"Δ vs flat={delta_pp:.2f} pp | win={win} | trainable={n_fourier:,}",
                flush=True,
            )

    wins = sum(1 for L in layers if fourier_auc[L] > flat_auc[L])
    deepest = max(layers)
    flat_deep_pp = (flat_auc[deepest] - classical_auc) * 100.0
    fourier_deep_pp = (fourier_auc[deepest] - classical_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_087 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        classical_val_auc=round(classical_auc, 6),
        flat_auc_by_layers={str(k): round(v, 6) for k, v in flat_auc.items()},
        fourier_auc_by_layers={str(k): round(v, 6) for k, v in fourier_auc.items()},
        rung_wins=wins,
        n_rungs=len(layers),
        flat_deep_vs_classical_pp=round(flat_deep_pp, 3),
        fourier_deep_vs_classical_pp=round(fourier_deep_pp, 3),
        elapsed_s=round(elapsed, 3),
    )

    return FourierReuploadResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        classical_val_auc=classical_auc,
        flat_auc_by_layers=flat_auc,
        fourier_auc_by_layers=fourier_auc,
        rung_wins=wins,
        n_rungs=len(layers),
        deepest_layers=deepest,
        flat_deep_vs_classical_pp=flat_deep_pp,
        fourier_deep_vs_classical_pp=fourier_deep_pp,
        min_rung_wins=min_wins,
        min_vs_classical_pp=min_vs_cl,
        n_trainable_flat=n_flat,
        n_trainable_fourier=n_fourier,
        elapsed_s=round(elapsed, 3),
        profile=profile,
    )


def _summarize(result: FourierReuploadResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    lines = [
        f"\n{'=' * 60}",
        "EXP 087 SUMMARY",
        f"{'=' * 60}",
        f"Train rows: {result.n_train_rows:,} | Val rows: {result.n_val_rows:,}",
        f"Classical AUC: {result.classical_val_auc:.4f}",
    ]
    for L in sorted(result.flat_auc_by_layers):
        f_auc = result.flat_auc_by_layers[L]
        q_auc = result.fourier_auc_by_layers[L]
        lines.append(
            f"L={L}: flat={f_auc:.4f} | fourier={q_auc:.4f} | "
            f"Δ={(q_auc - f_auc) * 100.0:.2f} pp"
        )
    lines.extend(
        [
            f"Rung wins: {result.rung_wins}/{result.n_rungs} "
            f"(gate ≥ {result.min_rung_wins})",
            f"Deepest flat Δ classical: {result.flat_deep_vs_classical_pp:.2f} pp",
            f"Deepest Fourier Δ classical: {result.fourier_deep_vs_classical_pp:.2f} pp",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def _build_results_md(result: FourierReuploadResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    rung_rows = []
    for L in sorted(result.flat_auc_by_layers):
        f_auc = result.flat_auc_by_layers[L]
        q_auc = result.fourier_auc_by_layers[L]
        delta = (q_auc - f_auc) * 100.0
        rung_rows.append(
            f"| {L} | {f_auc:.4f} | {q_auc:.4f} | {delta:+.2f} | "
            f"{'win' if q_auc > f_auc else 'lose'} |"
        )
    return "\n".join(
        [
            "# Results — EXP 087: Fourier re-upload vs flat angle head (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Train rows: **{result.n_train_rows:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Classical distill ResidualNano AUC: **{result.classical_val_auc:.4f}**",
            f"- Rung wins (Fourier > flat): **{result.rung_wins}/{result.n_rungs}** "
            f"(gate ≥ {result.min_rung_wins})",
            f"- Deepest flat Δ classical: **{result.flat_deep_vs_classical_pp:.2f} pp**",
            f"- Deepest Fourier Δ classical: **{result.fourier_deep_vs_classical_pp:.2f} pp**",
            f"- Parity floor: ≥ classical − {abs(result.min_vs_classical_pp)} pp",
            f"- Trainable (deepest flat / Fourier): "
            f"{result.n_trainable_flat:,} / {result.n_trainable_fourier:,}",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Per-rung AUC",
            "",
            "| Layers | Flat | Fourier | Δ pp | Result |",
            "|--------|------|---------|------|--------|",
            *rung_rows,
            "",
            "## Verdict",
            f"**{verdict}** — Phase B H-Q2.2 Fourier climate re-upload.",
            "",
            "## Limitations",
            "- Frozen distill backbone; PennyLane TorchLayer on CPU.",
            "- Hybrid fine-tune row budget capped for QNN wall-time on 4060.",
            "- Agro research benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 087 — Fourier re-upload on maize")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_087(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
