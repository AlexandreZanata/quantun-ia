"""
EXP 080 — Quantum champion fusion on ACYD (frozen C4 + warm-start + noise + GV-ALR).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_080_quantum_champion_fusion_acyd/run.py --profile publication --write-results
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

from src.classical.large_nano_mlp import LargeNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.quantum.large_nano_hybrid import LargeNanoHybrid
from src.training.adaptive_lr import AdaptiveLRConfig
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.quantum_warmstart import (
    WarmStartConfig,
    train_large_nano_hybrid_warmstart_adaptive,
)
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_080_quantum_champion_fusion_acyd"
EXP_ID = "exp_080"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ChampionFusionAcydResult:
    n_train_rows: int
    n_val_rows: int
    n_trainable_params: int
    classical_val_auc: float
    hybrid_baseline_val_auc: float
    champion_val_auc: float
    vs_classical_pp: float
    vs_best_hybrid_pp: float
    min_vs_classical_pp: float
    min_vs_best_hybrid_pp: float
    classical_epochs: int
    quantum_epochs: int
    depolarizing_p: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _load_backbone_checkpoint(cfg: dict, root: Path) -> dict[str, torch.Tensor]:
    exp_id = str(cfg.get("checkpoint_exp_id", "exp_060"))
    model_name = str(cfg.get("checkpoint_model_name", "large_nano_mlp"))
    seed = int(cfg.get("seed", 42))
    weights_path = root / "artifacts" / exp_id / model_name / f"seed_{seed}" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"backbone checkpoint missing at {weights_path} — run make exp-060-publication first"
        )
    return torch.load(weights_path, map_location="cpu", weights_only=True)


def _training_uses_cuda() -> bool:
    return os.environ.get("QML_DEVICE", "auto").lower() == "cuda" and torch.cuda.is_available()


def _build_hybrid(
    input_dim: int,
    cfg: dict,
    state_dict: dict[str, torch.Tensor],
    *,
    depolarizing_p: float,
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
        depolarizing_p=depolarizing_p,
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


def _eval_auc(model: torch.nn.Module, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
    return float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])


def _eval_classical(
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
    return _eval_auc(classical, x_val.to(device), y_val.to(device))


def _eval_champion_noiseless(
    champion: LargeNanoHybrid,
    *,
    input_dim: int,
    cfg: dict,
    backbone_state: dict[str, torch.Tensor],
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> float:
    """Eval with p=0 head so serve path matches noiseless LargeNanoHybrid."""
    serve_model = _build_hybrid(input_dim, cfg, copy.deepcopy(backbone_state), depolarizing_p=0.0)
    head_state = champion.export_noiseless_head_state()
    missing, unexpected = serve_model.load_state_dict(head_state, strict=False)
    unexpected_non_bb = [k for k in unexpected if not k.startswith("backbone.")]
    if unexpected_non_bb:
        raise RuntimeError(f"unexpected head keys when exporting noiseless: {unexpected_non_bb}")
    # Backbone already loaded; ignore missing backbone keys from head-only state.
    _ = missing
    serve_model.eval()
    return _eval_auc(serve_model, x_val, y_val)


def gate_passed(result: ChampionFusionAcydResult) -> bool:
    return (
        result.vs_classical_pp >= result.min_vs_classical_pp
        and result.vs_best_hybrid_pp >= result.min_vs_best_hybrid_pp
    )


def run_exp_080(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ChampionFusionAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train = _resolve_row_cap(cfg.get("n_train_rows"), 50_107)
    n_val = _resolve_row_cap(cfg.get("n_val_rows"), 5_830)
    depolarizing_p = float(cfg.get("depolarizing_p", 0.03))
    total_head_epochs = int(cfg.get("total_head_epochs", 8))
    classical_fraction = float(cfg.get("classical_fraction", 0.375))
    baseline_epochs = int(cfg.get("baseline_epochs", total_head_epochs))
    min_vs_classical_pp = float(cfg.get("min_vs_classical_pp", -1.0))
    min_vs_best_hybrid_pp = float(cfg.get("min_vs_best_hybrid_pp", 0.5))
    best_hybrid_ref = cfg.get("best_hybrid_ref_auc")

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 080 — Quantum champion fusion (ACYD) | profile={profile} | "
            f"p={depolarizing_p} | epochs={total_head_epochs}"
        )
        print(
            f"Gates: vs classical ≥ {min_vs_classical_pp} pp · "
            f"vs best hybrid ≥ {min_vs_best_hybrid_pp} pp"
        )
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

    backbone_state = _load_backbone_checkpoint(cfg, ROOT)
    classical_auc = _eval_classical(
        backbone_state,
        input_dim=input_dim,
        cfg=cfg,
        x_val=x_val_t,
        y_val=y_val_t,
    )
    if verbose:
        print(f"Classical C4 val ROC-AUC={classical_auc:.4f}", flush=True)

    train_kwargs = dict(
        batch_size=int(cfg.get("batch_size", 512)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )

    if best_hybrid_ref is not None and float(best_hybrid_ref) > 0 and profile == "publication":
        hybrid_baseline_auc = float(best_hybrid_ref)
        if verbose:
            print(
                f"Using curated best-hybrid ref AUC={hybrid_baseline_auc:.4f} (exp_065 fixed)",
                flush=True,
            )
    else:
        if verbose:
            print("Training noiseless hybrid baseline...", flush=True)
        baseline = _build_hybrid(
            input_dim, cfg, copy.deepcopy(backbone_state), depolarizing_p=0.0
        )
        train_model_batched(
            baseline,
            x_train_t,
            y_train_t,
            EXP_ID,
            "hybrid_baseline",
            epochs=baseline_epochs,
            lr=float(cfg.get("learning_rate", 0.01)),
            **train_kwargs,
        )
        hybrid_baseline_auc = _eval_auc(baseline, x_val_t, y_val_t)

    if verbose:
        print("Training champion fusion (warm-start + noise + GV-ALR)...", flush=True)
    champion = _build_hybrid(
        input_dim,
        cfg,
        copy.deepcopy(backbone_state),
        depolarizing_p=depolarizing_p,
    )
    n_trainable = sum(p.numel() for p in champion.parameters() if p.requires_grad)
    ws_cfg = WarmStartConfig(
        classical_fraction=classical_fraction,
        total_epochs=total_head_epochs,
    )
    classical_epochs, quantum_epochs = train_large_nano_hybrid_warmstart_adaptive(
        champion,
        x_train_t,
        y_train_t,
        EXP_ID,
        "quantum_nano_champion_acyd",
        config=ws_cfg,
        adaptive_config=_adaptive_config(cfg),
        lr=float(cfg.get("learning_rate", 0.01)),
        batch_size=int(cfg.get("batch_size", 512)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        x_val=x_val_t,
        y_val=y_val_t,
        seed=seed,
        profile=profile,
        save_checkpoints=bool(cfg.get("save_checkpoints", False)),
    )
    champion_auc = _eval_champion_noiseless(
        champion,
        input_dim=input_dim,
        cfg=cfg,
        backbone_state=backbone_state,
        x_val=x_val_t,
        y_val=y_val_t,
    )

    vs_classical_pp = (champion_auc - classical_auc) * 100.0
    vs_best_hybrid_pp = (champion_auc - hybrid_baseline_auc) * 100.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_080 champion fusion summary",
        exp_id=EXP_ID,
        profile=profile,
        classical_val_auc=round(classical_auc, 4),
        hybrid_baseline_val_auc=round(hybrid_baseline_auc, 4),
        champion_val_auc=round(champion_auc, 4),
        vs_classical_pp=round(vs_classical_pp, 3),
        vs_best_hybrid_pp=round(vs_best_hybrid_pp, 3),
        classical_epochs=classical_epochs,
        quantum_epochs=quantum_epochs,
        elapsed_s=round(elapsed, 3),
    )

    result = ChampionFusionAcydResult(
        n_train_rows=len(y_train),
        n_val_rows=len(y_val),
        n_trainable_params=n_trainable,
        classical_val_auc=classical_auc,
        hybrid_baseline_val_auc=hybrid_baseline_auc,
        champion_val_auc=champion_auc,
        vs_classical_pp=vs_classical_pp,
        vs_best_hybrid_pp=vs_best_hybrid_pp,
        min_vs_classical_pp=min_vs_classical_pp,
        min_vs_best_hybrid_pp=min_vs_best_hybrid_pp,
        classical_epochs=classical_epochs,
        quantum_epochs=quantum_epochs,
        depolarizing_p=depolarizing_p,
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"classical={classical_auc:.4f} | baseline={hybrid_baseline_auc:.4f} | "
            f"champion={champion_auc:.4f} | vs_C4={vs_classical_pp:+.2f} pp | "
            f"vs_hybrid={vs_best_hybrid_pp:+.2f} pp [{status}] | {elapsed:.1f}s",
            flush=True,
        )

    return result


def _summarize(result: ChampionFusionAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 080 SUMMARY",
            f"{'=' * 60}",
            f"Train: {result.n_train_rows:,} | Val: {result.n_val_rows:,}",
            f"Schedule: {result.classical_epochs} classical + {result.quantum_epochs} GV-ALR "
            f"(p={result.depolarizing_p})",
            f"Classical C4 ROC-AUC: {result.classical_val_auc:.4f}",
            f"Best hybrid baseline: {result.hybrid_baseline_val_auc:.4f}",
            f"Champion (noiseless eval): {result.champion_val_auc:.4f}",
            f"vs classical: {result.vs_classical_pp:+.2f} pp (gate ≥ {result.min_vs_classical_pp})",
            f"vs best hybrid: {result.vs_best_hybrid_pp:+.2f} pp "
            f"(gate ≥ {result.min_vs_best_hybrid_pp})",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: ChampionFusionAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 080: Quantum champion fusion on ACYD (C4)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gates (ROC-AUC)",
            "",
            f"- Train rows: **{result.n_train_rows:,}** · Val rows: **{result.n_val_rows:,}**",
            f"- Trainable head params: **{result.n_trainable_params:,}**",
            f"- Schedule: **{result.classical_epochs}** warm-start + **{result.quantum_epochs}** "
            f"GV-ALR · depolarizing p=**{result.depolarizing_p}** (train-only)",
            f"- Classical C4: **{result.classical_val_auc:.4f}**",
            f"- Best hybrid baseline: **{result.hybrid_baseline_val_auc:.4f}**",
            f"- Champion (noiseless eval): **{result.champion_val_auc:.4f}**",
            f"- vs classical: **{result.vs_classical_pp:+.2f} pp**",
            f"- vs best hybrid: **{result.vs_best_hybrid_pp:+.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — parity vs C4 (≥ {result.min_vs_classical_pp} pp) and lift vs best "
            f"frozen hybrid (≥ {result.min_vs_best_hybrid_pp} pp).",
            "",
            "## Limitations",
            "- Train-time noise on `default.mixed`; eval copies weights to noiseless head.",
            "- Entangle / angle-encoding / re-upload depth curriculum excluded from fusion.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 080 — quantum champion fusion on ACYD (C4)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_080(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
