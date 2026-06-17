#!/usr/bin/env python3
"""CLI for Nano Parity Bench — quantum nanomodel vs parameter-matched classical."""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.application.dto import NanoParityBenchDTO
from src.application.nano_parity_bench import execute, run_suite
from src.application.parity_datasets import ensure_datasets_available
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qml-bench-parity",
        description=(
            "Fair holdout comparison: quantum nanomodel vs parameter-matched classical MLP. "
            "Downloads UCI/MNIST data automatically."
        ),
    )
    parser.add_argument("--model", help="Quantum nanomodel (e.g. hybrid_sandwich)")
    parser.add_argument("--dataset", help="Tabular dataset (e.g. wine_binary)")
    parser.add_argument("--profile", default="ci", help="Profile: ci, publication")
    parser.add_argument("--exp-id", default="exp_022", dest="exp_id")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--suite", action="store_true", help="Run full suite from config")
    parser.add_argument("--download-only", action="store_true", help="Cache datasets and exit")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")

    if args.download_only:
        status = ensure_datasets_available(
            ["breast_cancer", "wine_binary", "iris_binary", "mnist_binary"]
        )
        print(json.dumps({"datasets": status}, indent=2))
        return 0

    if args.suite:
        results = run_suite(profile=args.profile, exp_id=args.exp_id)
        payload = [_result_payload(r) for r in results]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in payload:
                _print_result(item)
        return 0

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required unless --suite or --download-only")

    dto = NanoParityBenchDTO(
        quantum_model=args.model,
        dataset=args.dataset,
        profile=args.profile,
        exp_id=args.exp_id,
        epochs=args.epochs,
    )
    outcome = execute(dto)
    if isinstance(outcome, Fail):
        print(f"Error [{outcome.error.code}]: {outcome.error.message}", file=sys.stderr)
        return 1

    assert isinstance(outcome, Ok)
    payload = _result_payload(outcome.value)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_result(payload)
    return 0


def _result_payload(result) -> dict:
    comp = result.comparison
    return {
        "exp_id": result.exp_id,
        "quantum_model": result.quantum_model,
        "dataset": result.dataset,
        "profile": result.profile,
        "classical_baseline": result.classical_label,
        "quantum_n_params": result.quantum_n_params,
        "classical_n_params": result.classical_n_params,
        "param_delta": result.param_delta,
        "quantum_mean_accuracy": result.quantum_mean,
        "classical_mean_accuracy": result.classical_mean,
        "mean_diff_pp": round(comp.get("mean_diff", 0.0) * 100, 2),
        "p_value": comp.get("p_value"),
        "significant": comp.get("significant"),
        "cohens_d": comp.get("effect_size_cohens_d"),
        "effect_magnitude": comp.get("effect_size_magnitude"),
        "quantum_wins": result.quantum_wins,
        "verdict": result.verdict,
        "datasets_status": result.datasets_status,
    }


def _print_result(payload: dict) -> None:
    print(f"=== {payload['quantum_model']} vs {payload['classical_baseline']} on {payload['dataset']} ===")
    print(
        f"Params: quantum={payload['quantum_n_params']} classical={payload['classical_n_params']} "
        f"(Δ={payload['param_delta']})"
    )
    q_pct = payload["quantum_mean_accuracy"] * 100
    c_pct = payload["classical_mean_accuracy"] * 100
    print(f"Holdout: quantum={q_pct:.1f}% classical={c_pct:.1f}% (Δ={payload['mean_diff_pp']:+.1f} pp)")
    print(
        f"Wilcoxon p={payload['p_value']} | Cohen's d={payload['cohens_d']} "
        f"({payload['effect_magnitude']}) | verdict={payload['verdict']}"
    )
    if payload["quantum_wins"]:
        print("✓ Quantum nanomodel significantly outperforms matched classical at equal param budget.")
    else:
        print("✗ Claim not established — see verdict above.")


if __name__ == "__main__":
    raise SystemExit(main())
