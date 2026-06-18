#!/usr/bin/env python3
"""Demo — train QuantumNano-BC on breast cancer and predict sample rows."""

from __future__ import annotations

import argparse
import json
import os
import sys

from sklearn.datasets import load_breast_cancer

from src.application.dto import PredictNanomodelDTO, TrainNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.application.train_nanomodel import execute as train_execute
from src.shared.result import Fail, Ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train hybrid_sandwich and run inference demo")
    parser.add_argument("--profile", default="publication")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-id", default="quantum_nano_bc_app", dest="exp_id")
    parser.add_argument("--rows", type=int, default=5, help="Number of holdout rows to score")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    train_dto = TrainNanomodelDTO(
        model_name="hybrid_sandwich",
        dataset="breast_cancer",
        profile=args.profile,
        epochs=args.epochs,
        seed=args.seed,
        exp_id=args.exp_id,
        save_checkpoints=True,
    )
    print(f"Training {train_dto.model_name} × {train_dto.dataset} (profile={train_dto.profile})...")
    train_outcome = train_execute(train_dto)
    if isinstance(train_outcome, Fail):
        print(f"Train failed: {train_outcome.error.message}", file=sys.stderr)
        return 1

    assert isinstance(train_outcome, Ok)
    tr = train_outcome.value
    print(f"Holdout accuracy: {tr.accuracy * 100:.2f}% | checkpoint: {tr.checkpoint_path}")

    raw = load_breast_cancer()
    sample_rows = raw.data[: args.rows].astype(float).tolist()
    pred_outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=args.exp_id,
            model_name="hybrid_sandwich",
            dataset="breast_cancer",
            seed=args.seed,
            features=sample_rows,
        )
    )
    if isinstance(pred_outcome, Fail):
        print(f"Predict failed: {pred_outcome.error.message}", file=sys.stderr)
        return 1

    assert isinstance(pred_outcome, Ok)
    pr = pred_outcome.value
    print(json.dumps({"probabilities": pr.probabilities, "labels": pr.labels}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
