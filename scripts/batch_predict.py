#!/usr/bin/env python3
"""Batch score tabular features from CSV/JSON using saved nanomodel checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.application.batch_predict import (
    BatchPredictDTO,
    load_input_rows,
    run_batch_predict,
    write_output_csv,
    write_output_json,
)
from src.shared.result import Fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch predict probabilities from CSV/JSON")
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--exp-id", default="quantum_nano_bc_app", dest="exp_id")
    parser.add_argument("--model", default="hybrid_sandwich", dest="model_name")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=64, dest="chunk_size")
    parser.add_argument("--json", action="store_true", help="Write JSON output instead of CSV")
    args = parser.parse_args(argv)

    os.environ.setdefault("MLFLOW_DISABLE", "1")
    os.environ.setdefault("QML_DEVICE", "cuda")

    rows, _ = load_input_rows(args.input_path, dataset=args.dataset)
    dto = BatchPredictDTO(
        features=rows,
        exp_id=args.exp_id,
        model_name=args.model_name,
        dataset=args.dataset,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    outcome = run_batch_predict(dto)
    if isinstance(outcome, Fail):
        print(f"Error [{outcome.error.code}]: {outcome.error.message}", file=sys.stderr)
        return 1

    result = outcome.value
    source = str(args.input_path)
    if args.json or args.output_path.suffix.lower() == ".json":
        write_output_json(args.output_path, result, source_input=source)
    else:
        write_output_csv(args.output_path, result, source_input=source)

    print(
        f"Scored {result.n_rows} rows → {args.output_path} "
        f"(checkpoint={result.checkpoint_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
