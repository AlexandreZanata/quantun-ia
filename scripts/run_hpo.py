#!/usr/bin/env python3
"""CLI entry point for Optuna hyperparameter search."""

from __future__ import annotations

import argparse
import os

from src.training.hpo import build_exp_011_objective, run_optuna_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna HPO for an experiment")
    parser.add_argument("--exp", required=True, help="Experiment config key (e.g. exp_011_uci_tabular_qml)")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", default="quantum_angle", help="Model family for objective")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    objective = build_exp_011_objective(args.exp, args.profile, args.model)
    best = run_optuna_study(args.exp, objective, n_trials=args.trials, profile=args.profile)
    print("Best value:", best["best_value"])
    print("Best params:", best["best_params"])


if __name__ == "__main__":
    main()
