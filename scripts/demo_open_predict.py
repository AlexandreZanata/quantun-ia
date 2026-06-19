#!/usr/bin/env python3
"""Demo — evaluate trained serve checkpoint on open holdout data (RTX 4060)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.evaluate_serve_model import (
    SERVE_MODELS,
    EvaluateServeModelDTO,
    execute,
)
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained serve model on holdout split")
    parser.add_argument("--exp-id", default="exp_032")
    parser.add_argument("--model-name", default="large_nano_mlp")
    parser.add_argument("--dataset", default="higgs_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    if args.list_models:
        for m in SERVE_MODELS:
            print(f"{m.label}: exp_id={m.exp_id} model={m.model_name} dataset={m.dataset}")
        return 0

    outcome = execute(
        EvaluateServeModelDTO(
            exp_id=args.exp_id,
            model_name=args.model_name,
            dataset=args.dataset,
            seed=args.seed,
            split=args.split,
            n_rows=args.rows,
        )
    )
    if isinstance(outcome, Fail):
        print(f"Error [{outcome.error.code}]: {outcome.error.message}", file=sys.stderr)
        return 1

    assert isinstance(outcome, Ok)
    r = outcome.value
    summary = {
        "exp_id": r.exp_id,
        "model_name": r.model_name,
        "dataset": r.dataset,
        "split": r.split,
        "n_rows": r.n_rows,
        "roc_auc": round(r.roc_auc, 4),
        "accuracy": round(r.accuracy, 4),
        "brier_score": round(r.brier_score, 4),
        "confusion": {
            "tn": r.confusion.true_negative,
            "fp": r.confusion.false_positive,
            "fn": r.confusion.false_negative,
            "tp": r.confusion.true_positive,
        },
        "checkpoint_path": r.checkpoint_path,
        "sample_rows": r.sample_rows[:5],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
