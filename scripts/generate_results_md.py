#!/usr/bin/env python3
"""Write results.md for an experiment from logs/experiments.jsonl summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.training.results_writer import write_results_md

ROOT = Path(__file__).resolve().parents[1]

EXP_META: dict[str, dict[str, str]] = {
    "exp_011": {
        "dir": "exp_011_uci_tabular_qml",
        "title": "EXP 011 (UCI Tabular QML)",
        "dataset": "breast_cancer (UCI), 30% holdout",
        "conclusion": (
            "Compare perceptron, parameter-matched MLP, and angle-encoding QNN on UCI tabular data. "
            "See `docs/baselines.md` for literature context."
        ),
    },
    "exp_012": {
        "dir": "exp_012_mnist_pca_qml",
        "title": "EXP 012 (MNIST PCA QML)",
        "dataset": "MNIST 0 vs 1, PCA-8, 30% holdout",
        "conclusion": (
            "Angle vs amplitude encoding on PCA-reduced MNIST at matched qubit budget."
        ),
    },
    "exp_013": {
        "dir": "exp_013_augmentation_robustness",
        "title": "EXP 013 (Augmentation Robustness)",
        "dataset": "noisy circles, Gaussian augmentation",
        "conclusion": (
            "Gaussian augmentation vs baseline QNN on noisy circles."
        ),
    },
    "exp_014": {
        "dir": "exp_014_sequence_baselines",
        "title": "EXP 014 (Sequence Baselines)",
        "dataset": "sequential_binary synthetic",
        "conclusion": (
            "RNN-mini and Transformer-mini vs flattened QNN on sequential task."
        ),
    },
    "exp_015": {
        "dir": "exp_015_adaptive_qnn",
        "title": "EXP 015 (Adaptive QNN)",
        "dataset": "circles, noise=0.2, gradient-variance adaptive LR",
        "conclusion": (
            "Primary test: adaptive vs fixed LR at 6q×3l. See `docs/literature_review.md`."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate results.md from JSONL summaries")
    parser.add_argument(
        "--exp",
        nargs="+",
        default=list(EXP_META.keys()),
        help="Experiment ids (default: exp_011 through exp_015)",
    )
    parser.add_argument("--jsonl", type=Path, default=Path("logs/experiments.jsonl"))
    args = parser.parse_args()

    written: list[Path] = []
    for exp_id in args.exp:
        meta = EXP_META.get(exp_id)
        if meta is None:
            print(f"Unknown exp_id: {exp_id}", flush=True)
            return 1
        exp_dir = ROOT / "experiments" / meta["dir"]
        path = write_results_md(
            exp_id,
            exp_dir,
            exp_title=meta["title"],
            jsonl_path=args.jsonl,
            dataset_note=meta["dataset"],
            conclusion_hint=meta["conclusion"],
        )
        written.append(path)
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
