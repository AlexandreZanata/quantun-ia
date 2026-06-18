"""
EXP 030 — Publication large scale stability (30-seed hybrid on circles n=1000).

Run locally on RTX 4060:
  QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_030_publication_large/run.py --profile publication_large
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.generators import make_binary_classification
from src.data.splits import split_train_test
from src.quantum.hybrid_model import HybridSandwich
from src.training.config import load_experiment_config
from src.training.holdout import train_with_holdout

EXP_KEY = "exp_030_publication_large"
EXP_ID = "exp_030"
MODEL = "hybrid_sandwich"


@dataclass(frozen=True)
class ScaleStabilityResult:
    n_samples: int
    n_seeds: int
    reference_seeds: int
    mean_reference: float
    mean_all: float
    delta_pp: float
    parity_max_delta_pp: float
    per_seed_accuracies: tuple[float, ...]
    elapsed_s: float


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — run on RTX 4060 workstation with QML_DEVICE=cuda")


def _train_hybrid_seed(*, seed: int, cfg: dict) -> float:
    X, y, _ = make_binary_classification(
        n_samples=int(cfg["n_samples"]),
        dataset=str(cfg.get("dataset", "circles")),
        noise=float(cfg.get("noise", 0.2)),
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=float(cfg.get("test_size", 0.3)), random_state=seed
    )
    model_cfg = cfg.get("model_configs", {}).get(MODEL, {})
    n_qubits = int(model_cfg.get("n_qubits", cfg.get("n_qubits", 4)))
    n_layers = int(model_cfg.get("n_layers", cfg.get("n_layers", 2)))
    reupload = bool(model_cfg.get("reupload", True))
    lr = float(model_cfg.get("learning_rate", cfg.get("learning_rate", 0.02)))

    model = HybridSandwich(
        input_dim=2,
        n_qubits=n_qubits,
        n_layers=n_layers,
        reupload=reupload,
    )
    metrics = train_with_holdout(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        exp_id=EXP_ID,
        model_name=f"{MODEL}_seed{seed}",
        epochs=int(cfg["epochs"]),
        lr=lr,
        seed=seed,
        profile=cfg.get("profile"),
        save_checkpoints=False,
    )
    return float(metrics["accuracy"])


def run_exp_030(
    *,
    profile: str = "ci",
    verbose: bool = True,
    require_cuda: bool = True,
) -> ScaleStabilityResult:
    if require_cuda:
        _require_cuda()
        os.environ["QML_DEVICE"] = "cuda"
    os.environ.setdefault("MLFLOW_DISABLE", "1")

    cfg = load_experiment_config(EXP_KEY, profile=profile)
    seeds: list[int] = list(cfg["seeds"])
    ref_n = int(cfg.get("reference_seeds", 10))
    max_delta_pp = float(cfg.get("parity_max_delta_pp", 2.0))
    min_acc = float(cfg.get("min_seed_accuracy", 0.45))

    if ref_n < 1 or ref_n > len(seeds):
        raise ValueError(f"reference_seeds={ref_n} invalid for {len(seeds)} seeds")

    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"EXP 030 — Scale Stability | profile={profile} | "
            f"n={cfg['n_samples']} | seeds={len(seeds)} | ref={ref_n}"
        )
        print(f"Model: {MODEL} | epochs={cfg['epochs']} | gate≤{max_delta_pp} pp")
        print(f"{'=' * 60}\n")

    t0 = time.perf_counter()
    accuracies: list[float] = []
    for i, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"[{i}/{len(seeds)}] seed={seed} — training...", flush=True)
        acc = _train_hybrid_seed(seed=seed, cfg=cfg)
        accuracies.append(acc)
        if acc < min_acc:
            raise RuntimeError(f"seed {seed} holdout {acc * 100:.1f}% < {min_acc * 100:.0f}% floor")
        if verbose:
            print(f"  holdout={acc * 100:.2f}%", flush=True)

    elapsed = time.perf_counter() - t0
    ref_accs = accuracies[:ref_n]
    mean_ref = statistics.mean(ref_accs)
    mean_all = statistics.mean(accuracies)
    delta_pp = abs(mean_all - mean_ref) * 100.0

    if verbose:
        status = "OK" if delta_pp <= max_delta_pp else "FAIL"
        print(
            f"\nmean₁₀={mean_ref * 100:.2f}% mean₃₀={mean_all * 100:.2f}% "
            f"|Δ|={delta_pp:.2f} pp [{status}] elapsed={elapsed:.1f}s",
            flush=True,
        )

    return ScaleStabilityResult(
        n_samples=int(cfg["n_samples"]),
        n_seeds=len(seeds),
        reference_seeds=ref_n,
        mean_reference=mean_ref,
        mean_all=mean_all,
        delta_pp=delta_pp,
        parity_max_delta_pp=max_delta_pp,
        per_seed_accuracies=tuple(accuracies),
        elapsed_s=round(elapsed, 3),
    )


def _summarize(result: ScaleStabilityResult) -> str:
    accepted = result.delta_pp <= result.parity_max_delta_pp
    verdict = "accepted" if accepted else "rejected"
    return "\n".join(
        [
            f"\n{'=' * 60}",
            "EXP 030 SUMMARY",
            f"{'=' * 60}",
            f"Samples: {result.n_samples} | Seeds: {result.n_seeds} (ref={result.reference_seeds})",
            f"Mean (ref): {result.mean_reference * 100:.2f}%",
            f"Mean (all): {result.mean_all * 100:.2f}%",
            f"|Δmean|: {result.delta_pp:.2f} pp (gate ≤ {result.parity_max_delta_pp} pp)",
            f"Elapsed: {result.elapsed_s}s",
            f"Verdict: {verdict}",
            f"{'=' * 60}\n",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP 030 — publication large scale stability")
    parser.add_argument("--profile", default="ci", choices=["ci", "publication_large"])
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    result = run_exp_030(profile=args.profile, verbose=not args.quiet)
    print(_summarize(result))

    if args.write_results:
        out = Path(__file__).resolve().parent / "results.md"
        out.write_text(_build_results_md(result), encoding="utf-8")
        print(f"Wrote {out}")

    return 0 if result.delta_pp <= result.parity_max_delta_pp else 1


def _build_results_md(result: ScaleStabilityResult) -> str:
    from datetime import date

    verdict = "accepted" if result.delta_pp <= result.parity_max_delta_pp else "rejected"
    return "\n".join(
        [
            "# Results — EXP 030: Publication Large Scale Stability",
            "",
            f"**Run date:** {date.today().isoformat()}  ",
            "**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)",
            "",
            "## Scale stability",
            "",
            f"- n_samples: **{result.n_samples}**",
            f"- Seeds: **{result.n_seeds}** (reference: first **{result.reference_seeds}**)",
            f"- Mean (reference): **{result.mean_reference * 100:.2f}%**",
            f"- Mean (all): **{result.mean_all * 100:.2f}%**",
            f"- |Δmean|: **{result.delta_pp:.2f} pp**",
            f"- Elapsed: **{result.elapsed_s}s**",
            "",
            "## Verdict",
            f"**{verdict}** — 30-seed hybrid mean within {result.parity_max_delta_pp} pp of 10-seed reference.",
            "",
            "## Limitations",
            "- Circles synthetic data only; research prototype.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
