#!/usr/bin/env python3
"""CLI for Nano Trainer — mini real-data training sessions."""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.application.dto import TrainNanomodelDTO
from src.application.train_nanomodel import execute
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qml-train",
        description="Run a mini nanomodel training session on a real dataset",
    )
    parser.add_argument("--model", required=True, help="Nanomodel name (e.g. perceptron)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. breast_cancer)")
    parser.add_argument("--profile", default="mini", help="Profile: mini, ci, publication")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-id", default="nano_train", dest="exp_id")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")

    dto = TrainNanomodelDTO(
        model_name=args.model,
        dataset=args.dataset,
        profile=args.profile,
        epochs=args.epochs,
        seed=args.seed,
        exp_id=args.exp_id,
    )
    result = execute(dto)

    if isinstance(result, Fail):
        print(f"Error [{result.error.code}]: {result.error.message}", file=sys.stderr)
        return 1

    assert isinstance(result, Ok)
    r = result.value
    payload = {
        "exp_id": r.exp_id,
        "model": r.model_name,
        "dataset": r.dataset,
        "profile": r.profile,
        "seed": r.seed,
        "accuracy": r.accuracy,
        "loss": r.loss,
        "elapsed_s": r.elapsed_s,
        "n_params": r.n_params,
        "n_epochs": r.n_epochs,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print(f"Holdout accuracy: {r.accuracy * 100:.1f}%")
        print(f"Loss: {r.loss:.4f}")
        print(f"Elapsed: {r.elapsed_s}s | Params: {r.n_params} | Epochs: {r.n_epochs}")
        print(f"Logged to logs/experiments.jsonl (exp_id={r.exp_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
