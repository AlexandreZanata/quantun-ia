"""
EXP 057 — Parameter-shift vs autograd on deep re-upload QNN (breast cancer).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_057_param_shift_ablation/run.py --profile publication --write-results
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset_registry import prepare_dataset
from src.quantum.qnn_reupload import QuantumNetReupload
from src.training.config import load_experiment_config
from src.training.param_shift_ablation import (
    AUTOGRAD_METHOD,
    PARAM_SHIFT_METHOD,
    ParamShiftAblationResult,
    gate_passed,
    measure_reupload_gradient_variance,
    train_reupload_grad_method,
)
from src.training.structured_log import init_correlation_id, log_event

EXP_KEY = "exp_057_param_shift_ablation"
EXP_ID = "exp_057"


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _build_model(input_dim: int, cfg: dict, diff_method: str) -> QuantumNetReupload:
    return QuantumNetReupload(
        n_qubits=int(cfg.get("n_qubits", 4)),
        n_layers=int(cfg.get("n_layers", 3)),
        input_dim=input_dim,
        diff_method=diff_method,
    )


def run_exp_057(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ParamShiftAblationResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds = tuple(int(s) for s in cfg.get("seeds", [42]))
    epochs = int(cfg.get("epochs", 20))
    lr = float(cfg.get("learning_rate", 0.02))
    max_holdout_pp = float(cfg.get("max_holdout_pp", 1.0))
    min_variance_ratio = float(cfg.get("min_variance_ratio", 2.0))
    grad_samples = int(cfg.get("grad_samples", 10))
    n_qubits = int(cfg.get("n_qubits", 4))
    n_layers = int(cfg.get("n_layers", 3))

    init_correlation_id()
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 057 — Parameter-shift vs autograd | profile={profile} | "
            f"seeds={len(seeds)} | {n_qubits}q × {n_layers}L re-upload"
        )
        print(f"Gate: holdout ≤ {max_holdout_pp} pp · var ratio ≥ {min_variance_ratio}")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    autograd_accs: list[float] = []
    param_shift_accs: list[float] = []

    dataset_name = str(cfg.get("dataset", "breast_cancer"))
    for seed in seeds:
        x_train, x_holdout, y_train, y_holdout, meta = prepare_dataset(
            dataset_name,
            random_state=seed,
        )
        input_dim = int(meta["n_features"])
        x_train_t = torch.tensor(x_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        x_hold_t = torch.tensor(x_holdout, dtype=torch.float32)
        y_hold_t = torch.tensor(y_holdout, dtype=torch.float32)

        if verbose:
            print(f"Seed {seed}: autograd...", flush=True)
        autograd_model = _build_model(input_dim, cfg, AUTOGRAD_METHOD)
        autograd_acc = train_reupload_grad_method(
            autograd_model,
            x_train_t,
            y_train_t,
            x_hold_t,
            y_hold_t,
            EXP_ID,
            f"autograd_seed{seed}",
            epochs=epochs,
            lr=lr,
            seed=seed,
            profile=profile,
        )
        autograd_accs.append(autograd_acc)

        if verbose:
            print(f"Seed {seed}: parameter-shift...", flush=True)
        param_model = _build_model(input_dim, cfg, PARAM_SHIFT_METHOD)
        param_acc = train_reupload_grad_method(
            param_model,
            x_train_t,
            y_train_t,
            x_hold_t,
            y_hold_t,
            EXP_ID,
            f"param_shift_seed{seed}",
            epochs=epochs,
            lr=lr,
            seed=seed,
            profile=profile,
        )
        param_shift_accs.append(param_acc)

        if verbose:
            print(
                f"Seed {seed}: autograd={autograd_acc:.4f} param-shift={param_acc:.4f} "
                f"Δ={(autograd_acc - param_acc) * 100:+.2f} pp",
                flush=True,
            )

    mean_autograd = statistics.mean(autograd_accs)
    mean_param_shift = statistics.mean(param_shift_accs)
    mean_holdout_pp = abs(mean_autograd - mean_param_shift) * 100.0

    if verbose:
        print("Measuring gradient variance...", flush=True)

    x_train, _x_hold, _y_train, _y_hold, meta = prepare_dataset(dataset_name, random_state=seeds[0])
    input_dim = int(meta["n_features"])
    grad_seed = int(cfg.get("seed", 42))

    autograd_var = measure_reupload_gradient_variance(
        n_qubits=n_qubits,
        n_layers=n_layers,
        input_dim=input_dim,
        diff_method=AUTOGRAD_METHOD,
        n_samples=grad_samples,
        seed=grad_seed,
    )
    param_var = measure_reupload_gradient_variance(
        n_qubits=n_qubits,
        n_layers=n_layers,
        input_dim=input_dim,
        diff_method=PARAM_SHIFT_METHOD,
        n_samples=grad_samples,
        seed=grad_seed,
    )
    variance_ratio = autograd_var / param_var if param_var > 0 else 0.0
    elapsed = time.perf_counter() - t0

    log_event(
        "info",
        "exp_057 param-shift ablation summary",
        exp_id=EXP_ID,
        profile=profile,
        mean_autograd_acc=round(mean_autograd, 4),
        mean_param_shift_acc=round(mean_param_shift, 4),
        mean_holdout_pp=round(mean_holdout_pp, 3),
        autograd_grad_var=round(autograd_var, 6),
        param_shift_grad_var=round(param_var, 6),
        variance_ratio=round(variance_ratio, 3),
    )

    result = ParamShiftAblationResult(
        n_seeds=len(seeds),
        mean_autograd_acc=round(mean_autograd, 4),
        mean_param_shift_acc=round(mean_param_shift, 4),
        mean_holdout_pp=round(mean_holdout_pp, 3),
        autograd_grad_var=round(autograd_var, 6),
        param_shift_grad_var=round(param_var, 6),
        variance_ratio=round(variance_ratio, 3),
        max_holdout_pp=max_holdout_pp,
        min_variance_ratio=min_variance_ratio,
        elapsed_s=round(elapsed, 3),
    )

    if verbose:
        status = "OK" if gate_passed(result) else "FAIL"
        print(
            f"\nHoldout Δ={mean_holdout_pp:.2f} pp | var ratio={variance_ratio:.2f} [{status}] "
            f"| elapsed={elapsed:.1f}s",
            flush=True,
        )

    return result


def _summarize(result: ParamShiftAblationResult) -> str:
    verdict = "accepted" if gate_passed(result) else "honest_negative"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 057 SUMMARY",
            f"{'=' * 60}",
            f"Mean autograd holdout: {result.mean_autograd_acc:.4f}",
            f"Mean param-shift holdout: {result.mean_param_shift_acc:.4f}",
            f"Holdout gap: {result.mean_holdout_pp:.2f} pp (gate ≤ {result.max_holdout_pp})",
            f"Grad var autograd: {result.autograd_grad_var:.6f}",
            f"Grad var param-shift: {result.param_shift_grad_var:.6f}",
            f"Variance ratio: {result.variance_ratio:.2f} (gate ≥ {result.min_variance_ratio})",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def _build_results_md(result: ParamShiftAblationResult) -> str:
    from datetime import date

    verdict = "accepted" if gate_passed(result) else "honest negative"
    return "\n".join(
        [
            "# Results — EXP 057: Parameter-shift vs autograd ablation",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Validation gate",
            "",
            f"- Mean holdout gap: **{result.mean_holdout_pp:.2f} pp** (gate ≤ **{result.max_holdout_pp}**)",
            f"- Variance ratio (autograd/param-shift): **{result.variance_ratio:.2f}** "
            f"(gate ≥ **{result.min_variance_ratio}**)",
            "",
            "| Method | Mean holdout acc | Grad variance |",
            "|--------|------------------|---------------|",
            f"| autograd | {result.mean_autograd_acc:.4f} | {result.autograd_grad_var:.6f} |",
            f"| parameter-shift | {result.mean_param_shift_acc:.4f} | {result.param_shift_grad_var:.6f} |",
            "",
            f"- Seeds: **{result.n_seeds}**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — parameter-shift vs autograd on 4q×3L re-upload QNN.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 057 — parameter-shift vs autograd ablation")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_057(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if gate_passed(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
