"""
EXP 067 — Re-upload depth curriculum on ACYD climate feature-block ladder (H-Q11).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_067_reupload_ladder_acyd/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.open_parquet import load_open_parquet_splits
from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.config import load_experiment_config
from src.training.reupload_curriculum import train_fixed_reupload, train_reupload_curriculum
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_067_reupload_ladder_acyd"
EXP_ID = "exp_067"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RungResult:
    rung_id: str
    metric_name: str
    feature_slice: tuple[int, int]
    n_features: int
    curriculum_score: float
    fixed_score: float
    advantage_pp: float
    won: bool


@dataclass(frozen=True)
class ReuploadLadderAcydResult:
    n_rungs: int
    n_wins: int
    min_wins: int
    min_rung_advantage_pp: float
    rung_results: tuple[RungResult, ...]
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _build_model(input_dim: int, cfg: dict, n_layers: int) -> QuantumNetReupload:
    return QuantumNetReupload(
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=n_layers,
        input_dim=input_dim,
    )


def _slice_features(
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_slice: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    start, end = int(feature_slice[0]), int(feature_slice[1])
    return x_train[:, start:end], x_val[:, start:end], (start, end)


def gate_passed(result: ReuploadLadderAcydResult) -> bool:
    return result.n_wins >= result.min_wins


def _run_rung(
    rung: dict,
    *,
    cfg: dict,
    seed: int,
    profile: str,
    min_rung_advantage_pp: float,
    x_train_full: np.ndarray,
    y_train: np.ndarray,
    x_val_full: np.ndarray,
    y_val: np.ndarray,
    verbose: bool,
) -> RungResult:
    rung_id = str(rung["id"])
    metric = str(rung.get("metric", "roc_auc"))
    use_batched = bool(rung.get("use_batched", True))
    feature_slice = rung.get("feature_slice", [0, x_train_full.shape[1]])
    n_train = _resolve_row_cap(rung.get("n_train_rows"), len(y_train))
    n_val = _resolve_row_cap(rung.get("n_val_rows"), len(y_val))

    x_tr = x_train_full if n_train is None else x_train_full[:n_train]
    y_tr = y_train if n_train is None else y_train[:n_train]
    x_va = x_val_full if n_val is None else x_val_full[:n_val]
    y_va = y_val if n_val is None else y_val[:n_val]

    x_tr_s, x_va_s, slice_bounds = _slice_features(x_tr, x_va, feature_slice)
    input_dim = int(x_tr_s.shape[1])
    x_hold_t = torch.tensor(x_va_s, dtype=torch.float32)
    y_hold_t = torch.tensor(y_va, dtype=torch.float32)

    layer_ladder = tuple(int(x) for x in cfg.get("layer_ladder", [1, 2, 3]))
    max_layers = int(cfg.get("max_layers", layer_ladder[-1]))
    n_stages = int(cfg.get("curriculum_stages", len(layer_ladder)))
    epochs_per_stage = int(cfg.get("epochs_per_stage", 8))
    lr = float(cfg.get("learning_rate", 0.02))
    batch_size = int(cfg.get("batch_size", 512))
    weight_decay = float(cfg.get("weight_decay", 1e-4))

    if verbose:
        print(
            f"Rung {rung_id}: features[{slice_bounds[0]}:{slice_bounds[1]}] "
            f"dim={input_dim} curriculum...",
            flush=True,
        )
    curriculum_model = _build_model(input_dim, cfg, layer_ladder[0])
    curriculum_score = train_reupload_curriculum(
        curriculum_model,
        x_tr_s,
        y_tr,
        x_hold_t,
        y_hold_t,
        EXP_ID,
        f"curriculum_{rung_id}_seed{seed}",
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        layer_ladder=layer_ladder,
        lr=lr,
        seed=seed,
        profile=profile,
        metric=metric,
        use_batched=use_batched,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    if verbose:
        print(f"Rung {rung_id}: fixed L={max_layers}...", flush=True)

    def build(n_layers: int) -> QuantumNetReupload:
        return _build_model(input_dim, cfg, n_layers)

    fixed_score = train_fixed_reupload(
        build,
        max_layers,
        x_tr_s,
        y_tr,
        x_hold_t,
        y_hold_t,
        EXP_ID,
        f"fixed_{rung_id}_seed{seed}",
        n_stages=n_stages,
        epochs_per_stage=epochs_per_stage,
        lr=lr,
        seed=seed,
        profile=profile,
        metric=metric,
        use_batched=use_batched,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    advantage_pp = (curriculum_score - fixed_score) * 100.0
    won = advantage_pp >= min_rung_advantage_pp
    if verbose:
        print(
            f"Rung {rung_id}: curriculum={curriculum_score:.4f} fixed={fixed_score:.4f} "
            f"Δ={advantage_pp:.2f} pp {'WIN' if won else 'loss'}",
            flush=True,
        )

    return RungResult(
        rung_id=rung_id,
        metric_name=metric,
        feature_slice=slice_bounds,
        n_features=input_dim,
        curriculum_score=curriculum_score,
        fixed_score=fixed_score,
        advantage_pp=advantage_pp,
        won=won,
    )


def run_exp_067(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ReuploadLadderAcydResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_soy_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    rungs = list(cfg.get("rungs", []))
    min_wins = int(cfg.get("min_wins", 2))
    min_rung_pp = float(cfg.get("min_rung_advantage_pp", 0.3))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 067 — Re-upload climate feature ladder (ACYD) | profile={profile} | "
            f"rungs={len(rungs)} | gate ≥ {min_wins} wins"
        )
        print(f"Per-rung advantage ≥ {min_rung_pp} pp ROC-AUC")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    # Load full ACYD once; per-rung row caps and column masks applied in _run_rung.
    max_train = max((_resolve_row_cap(r.get("n_train_rows"), 50_107) or 50_107) for r in rungs) if rungs else 50_107
    max_val = max((_resolve_row_cap(r.get("n_val_rows"), 5_830) or 5_830) for r in rungs) if rungs else 5_830
    x_train_full, y_train, x_val_full, y_val, _x_test, _y_test, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=max_train,
        n_val_rows=max_val,
        random_state=seed,
    )

    rung_results: list[RungResult] = []
    for rung in rungs:
        rung_results.append(
            _run_rung(
                rung,
                cfg=cfg,
                seed=seed,
                profile=profile,
                min_rung_advantage_pp=min_rung_pp,
                x_train_full=x_train_full,
                y_train=y_train,
                x_val_full=x_val_full,
                y_val=y_val,
                verbose=verbose,
            )
        )

    n_wins = sum(1 for r in rung_results if r.won)
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_067 reupload ladder summary",
        exp_id=EXP_ID,
        profile=profile,
        n_rungs=len(rung_results),
        n_wins=n_wins,
        min_wins=min_wins,
        rung_advantages_pp=[round(r.advantage_pp, 3) for r in rung_results],
    )

    result = ReuploadLadderAcydResult(
        n_rungs=len(rung_results),
        n_wins=n_wins,
        min_wins=min_wins,
        min_rung_advantage_pp=min_rung_pp,
        rung_results=tuple(rung_results),
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(f"\nWins: {n_wins}/{len(rung_results)} [{status}] | elapsed={elapsed:.1f}s", flush=True)

    return result


def _summarize(result: ReuploadLadderAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    lines = [
        f"\n{'=' * 60}",
        "EXP 067 SUMMARY",
        f"{'=' * 60}",
        f"Wins: {result.n_wins}/{result.n_rungs} (gate ≥ {result.min_wins})",
    ]
    for rung in result.rung_results:
        lines.append(
            f"  {rung.rung_id} [{rung.feature_slice[0]}:{rung.feature_slice[1]}] "
            f"({rung.metric_name}, dim={rung.n_features}): "
            f"curriculum={rung.curriculum_score:.4f} fixed={rung.fixed_score:.4f} "
            f"Δ={rung.advantage_pp:+.2f} pp {'WIN' if rung.won else 'loss'}"
        )
    lines.extend(
        [
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )
    return "\n".join(lines)


def _build_results_md(result: ReuploadLadderAcydResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest negative"
    lines = [
        "# Results — EXP 067: Re-upload climate feature-block ladder (ACYD / H-Q11)",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
        "**Dataset:** `acyd_soy_brazil_v1`",
        "",
        "## Validation gate (ROC-AUC)",
        "",
        f"- Wins: **{result.n_wins}/{result.n_rungs}** (gate ≥ **{result.min_wins}**)",
        f"- Per-rung advantage gate: **≥ {result.min_rung_advantage_pp} pp**",
        "",
        "| Rung | Slice | Dim | Curriculum | Fixed | Δ pp | Won |",
        "|------|-------|-----|------------|-------|------|-----|",
    ]
    for rung in result.rung_results:
        lines.append(
            f"| {rung.rung_id} | [{rung.feature_slice[0]}:{rung.feature_slice[1]}] | "
            f"{rung.n_features} | {rung.curriculum_score:.4f} | {rung.fixed_score:.4f} | "
            f"{rung.advantage_pp:+.2f} | {'yes' if rung.won else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — re-upload depth curriculum vs fixed max depth on climate feature blocks.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EXP 067 — re-upload climate feature-block ladder (ACYD)"
    )
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_067(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
