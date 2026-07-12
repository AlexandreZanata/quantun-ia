"""
EXP 085 — Sample-efficiency curves: HistGB vs ResidualNano (hard + distill) on ACYD maize.

Run on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_085_sample_efficiency_agro/run.py \\
    --profile publication --write-results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.classical.residual_nano_mlp import ResidualNanoMLP
from src.data.open_parquet import load_open_parquet_splits
from src.training.batched_trainer import evaluate_with_auc, train_model_batched
from src.training.config import load_experiment_config
from src.training.distillation import mix_hard_soft_targets, soft_targets_from_proba
from src.training.sample_efficiency import area_under_budget_curve, stratified_row_budget
from src.training.structured_log import init_correlation_id, log_event
from src.training.trainer import count_parameters

EXP_KEY = "exp_085_sample_efficiency_agro"
EXP_ID = "exp_085"
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BudgetPoint:
    fraction: float
    n_train_rows: int
    histgb_auc: float
    hard_nano_auc: float
    distill_nano_auc: float
    distill_beats_histgb: bool


@dataclass(frozen=True)
class SampleEfficiencyResult:
    profile: str
    n_val_rows: int
    n_params: int
    points: tuple[BudgetPoint, ...]
    distill_wins: int
    min_wins: int
    distill_aulc: float
    histgb_aulc: float
    elapsed_s: float


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _resolve_row_cap(value: int | None, total: int) -> int | None:
    if value is None or value <= 0:
        return None
    return min(int(value), total)


def _parse_fractions(raw: object) -> list[float]:
    if isinstance(raw, str):
        return [float(x.strip()) for x in raw.split(",") if x.strip()]
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    raise ValueError(f"unsupported fractions config: {raw!r}")


def _build_student(input_dim: int, cfg: dict) -> ResidualNanoMLP:
    return ResidualNanoMLP(
        input_dim,
        hidden=int(cfg.get("residual_hidden", 512)),
        n_blocks=int(cfg.get("residual_n_blocks", 3)),
        bottleneck=int(cfg.get("residual_bottleneck", 64)),
        dropout=float(cfg.get("dropout", 0.2)),
    )


def _train_nano(
    *,
    model: ResidualNanoMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: dict,
    profile: str,
    seed: int,
    model_name: str,
) -> float:
    train_model_batched(
        model,
        x_train,
        y_train,
        EXP_ID,
        model_name,
        epochs=int(cfg["epochs"]),
        lr=float(cfg.get("learning_rate", 0.001)),
        batch_size=min(int(cfg.get("batch_size", 2048)), max(len(y_train), 1)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        X_val=x_val,
        y_val=y_val,
        seed=seed,
        profile=profile,
        save_checkpoints=False,
    )
    return float(evaluate_with_auc(model, x_val, y_val)["roc_auc"])


def _gate_passed(result: SampleEfficiencyResult) -> bool:
    return result.distill_wins >= result.min_wins


def run_exp_085(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> SampleEfficiencyResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    dataset_id = str(cfg.get("dataset_id", "acyd_maize_brazil_v1"))
    seed = int(cfg.get("seed", 42))
    n_train_cap = _resolve_row_cap(cfg.get("n_train_rows"), 200_000)
    n_val_cap = _resolve_row_cap(cfg.get("n_val_rows"), 50_000)
    fractions = _parse_fractions(cfg.get("fractions", [0.01, 0.05, 0.2, 1.0]))
    min_wins = int(cfg.get("min_budget_wins", 2))
    hgb_max_iter = int(cfg.get("histgb_max_iter", 100))
    alpha = float(cfg.get("distill_alpha", 1.0))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 085 — Sample efficiency | profile={profile} | "
            f"fractions={fractions} | gate ≥ {min_wins} distill wins"
        )
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    x_train_full, y_train_full, x_val, y_val, _xt, _yt, _scaler = load_open_parquet_splits(
        dataset_id,
        ROOT,
        n_train_rows=n_train_cap,
        n_val_rows=n_val_cap,
        random_state=seed,
    )
    input_dim = int(x_train_full.shape[1])
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    probe = _build_student(input_dim, cfg)
    n_params = count_parameters(probe)
    del probe

    points: list[BudgetPoint] = []
    for frac in fractions:
        x_sub, y_sub = stratified_row_budget(
            x_train_full, y_train_full, fraction=frac, random_state=seed
        )
        histgb = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.1,
            max_iter=hgb_max_iter,
            random_state=seed,
        )
        histgb.fit(x_sub, y_sub)
        histgb_auc = float(roc_auc_score(y_val, histgb.predict_proba(x_val)[:, 1]))

        soft = soft_targets_from_proba(histgb.predict_proba(x_sub))
        y_distill = mix_hard_soft_targets(y_sub, soft, alpha=alpha)

        x_t = torch.tensor(x_sub, dtype=torch.float32)
        y_hard_t = torch.tensor(y_sub, dtype=torch.float32)
        y_dist_t = torch.tensor(y_distill, dtype=torch.float32)

        hard_auc = _train_nano(
            model=_build_student(input_dim, cfg),
            x_train=x_t,
            y_train=y_hard_t,
            x_val=x_val_t,
            y_val=y_val_t,
            cfg=cfg,
            profile=profile,
            seed=seed,
            model_name=f"hard_f{frac}",
        )
        distill_auc = _train_nano(
            model=_build_student(input_dim, cfg),
            x_train=x_t,
            y_train=y_dist_t,
            x_val=x_val_t,
            y_val=y_val_t,
            cfg=cfg,
            profile=profile,
            seed=seed,
            model_name=f"distill_f{frac}",
        )
        beats = distill_auc >= histgb_auc
        point = BudgetPoint(
            fraction=frac,
            n_train_rows=len(y_sub),
            histgb_auc=histgb_auc,
            hard_nano_auc=hard_auc,
            distill_nano_auc=distill_auc,
            distill_beats_histgb=beats,
        )
        points.append(point)
        if verbose:
            flag = "WIN" if beats else "lose"
            print(
                f"f={frac:.2f} n={len(y_sub):,} | HistGB={histgb_auc:.4f} | "
                f"hard={hard_auc:.4f} | distill={distill_auc:.4f} [{flag}]",
                flush=True,
            )

    wins = sum(1 for p in points if p.distill_beats_histgb)
    fracs = [p.fraction for p in points]
    distill_aulc = area_under_budget_curve(fracs, [p.distill_nano_auc for p in points])
    histgb_aulc = area_under_budget_curve(fracs, [p.histgb_auc for p in points])
    elapsed = time.perf_counter() - t0

    result = SampleEfficiencyResult(
        profile=profile,
        n_val_rows=len(y_val),
        n_params=n_params,
        points=tuple(points),
        distill_wins=wins,
        min_wins=min_wins,
        distill_aulc=distill_aulc,
        histgb_aulc=histgb_aulc,
        elapsed_s=round(elapsed, 3),
    )

    art_dir = ROOT / "artifacts" / EXP_ID
    art_dir.mkdir(parents=True, exist_ok=True)
    curves_path = art_dir / f"curves_{profile}.json"
    curves_path.write_text(
        json.dumps(
            {
                "exp_id": EXP_ID,
                "profile": profile,
                "dataset_id": dataset_id,
                "n_params": n_params,
                "points": [asdict(p) for p in points],
                "distill_wins": wins,
                "min_wins": min_wins,
                "distill_aulc": distill_aulc,
                "histgb_aulc": histgb_aulc,
                "elapsed_s": result.elapsed_s,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log_event(
        "info",
        "exp_085 gate summary",
        exp_id=EXP_ID,
        profile=profile,
        distill_wins=wins,
        min_wins=min_wins,
        distill_aulc=round(distill_aulc, 6),
        histgb_aulc=round(histgb_aulc, 6),
        elapsed_s=result.elapsed_s,
        curves_path=str(curves_path),
    )
    if verbose:
        print(f"Wrote {curves_path}", flush=True)

    return result


def _summarize(result: SampleEfficiencyResult) -> str:
    verdict = "accepted" if _gate_passed(result) else "rejected"
    lines = [
        f"\n{'=' * 60}",
        "EXP 085 SUMMARY",
        f"{'=' * 60}",
        f"Val rows: {result.n_val_rows:,} | params={result.n_params:,}",
        f"Distill wins: {result.distill_wins}/{len(result.points)} (gate ≥ {result.min_wins})",
        f"AULC distill={result.distill_aulc:.4f} | HistGB={result.histgb_aulc:.4f}",
        f"Elapsed: {result.elapsed_s}s",
        f"Verdict: {verdict}",
        f"{'=' * 60}\n",
    ]
    return "\n".join(lines)


def _build_results_md(result: SampleEfficiencyResult) -> str:
    from datetime import date

    verdict = "accepted" if _gate_passed(result) else "rejected"
    rows = "\n".join(
        f"| {p.fraction:.0%} | {p.n_train_rows:,} | {p.histgb_auc:.4f} | "
        f"{p.hard_nano_auc:.4f} | {p.distill_nano_auc:.4f} | "
        f"{'yes' if p.distill_beats_histgb else 'no'} |"
        for p in result.points
    )
    return "\n".join(
        [
            "# Results — EXP 085: Sample-efficiency curves (ACYD maize)",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            f"**Profile:** `{result.profile}`",
            "",
            "## Validation gate",
            "",
            f"- Distill wins: **{result.distill_wins}/{len(result.points)}** "
            f"(gate ≥ {result.min_wins})",
            f"- AULC distill: **{result.distill_aulc:.4f}** | HistGB: **{result.histgb_aulc:.4f}**",
            f"- Student params: **{result.n_params:,}** | Val rows: **{result.n_val_rows:,}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "| Budget | Train rows | HistGB | Hard nano | Distill nano | Distill ≥ HistGB |",
            "|--------|------------|--------|-----------|--------------|------------------|",
            rows,
            "",
            "## Verdict",
            f"**{verdict}** — Phase A/C H-N2 sample-efficiency (row-fraction proxy).",
            "",
            "## Limitations",
            "- Stratified **row** budgets (processed parquet has no crop-year column).",
            "- Single seed; temporal val fixed across budgets.",
            "- Agro-climate benchmark — not operational planting advice.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 085 — sample-efficiency agro curves")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_085(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if _gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
