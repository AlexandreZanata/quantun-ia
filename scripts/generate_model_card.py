#!/usr/bin/env python3
"""Generate model card for the QuantumNano-BC flagship nano model."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dashboard.benchmark_data import latest_multi_seed_summaries, load_records

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = ROOT / "logs" / "experiments.jsonl"
DEFAULT_OUT = ROOT / "model_cards" / "quantum_nano_bc.md"
EXP_ID = "exp_024"
FLAGSHIP_MODEL = "hybrid_sandwich"


def _load_records(jsonl_path: Path) -> list[dict]:
    if jsonl_path == DEFAULT_JSONL:
        return load_records()
    if not jsonl_path.exists():
        return []
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find_checkpoint(records: list[dict]) -> str | None:
    for record in reversed(records):
        if record.get("exp_id") != EXP_ID:
            continue
        ctx = record.get("context") or {}
        if ctx.get("msg") == "checkpoint saved" and FLAGSHIP_MODEL in str(ctx.get("model_name", "")):
            return str(ctx.get("path"))
    return None


def generate_quantum_nano_bc_card(*, jsonl_path: Path = DEFAULT_JSONL) -> str:
    """Build model card markdown from latest exp_024 summaries."""
    records = _load_records(jsonl_path)
    summaries = latest_multi_seed_summaries(records)
    summary = summaries.get(EXP_ID, {})
    hybrid = summary.get(FLAGSHIP_MODEL, {})
    logistic = summary.get("logistic_regression", {})
    xgb = summary.get("xgboost_shallow", {})

    if not hybrid:
        raise ValueError(f"No multi_seed_summary for {EXP_ID} / {FLAGSHIP_MODEL} in {jsonl_path}")

    ckpt = _find_checkpoint(records)
    today = date.today().isoformat()
    hybrid_mean = hybrid.get("mean", 0.0) * 100
    hybrid_ci_low = hybrid.get("ci_low", 0.0) * 100
    hybrid_ci_high = hybrid.get("ci_high", 0.0) * 100
    logistic_mean = logistic.get("mean", 0.0) * 100 if logistic else None
    xgb_mean = xgb.get("mean", 0.0) * 100 if xgb else None

    lines = [
        "---",
        "title: QuantumNano-BC",
        "language: en",
        "license: mit",
        "tags:",
        "  - quantum-machine-learning",
        "  - tabular-classification",
        "  - breast-cancer",
        "datasets:",
        "  - breast_cancer",
        "---",
        "",
        "# QuantumNano-BC — Hybrid QML Nano Model",
        "",
        f"**Generated:** {today}  ",
        f"**Experiment:** `{EXP_ID}` (QuantumNano-BC flagship)  ",
        "**Architecture:** `hybrid_sandwich` (4 qubits, 2 re-upload layers)  ",
        "",
        "## Intended use",
        "",
        "Research benchmark for holdout-fair comparison of hybrid quantum–classical classifiers",
        "on the Wisconsin Breast Cancer (UCI) dataset. **Not for clinical deployment.**",
        "",
        "## Training data",
        "",
        "- **Dataset:** Wisconsin Breast Cancer (`sklearn.datasets.load_breast_cancer`)",
        "- **Samples:** 569 (full dataset, no subsampling in publication profile)",
        "- **Features:** 30 diagnostic measurements",
        "- **Split:** 30% stratified holdout before `StandardScaler` (train-fit only)",
        "",
        "## Evaluation results",
        "",
        "| Model | Mean holdout accuracy | 95% CI |",
        "|-------|----------------------|--------|",
        f"| **hybrid_sandwich** | {hybrid_mean:.1f}% | [{hybrid_ci_low:.1f}%, {hybrid_ci_high:.1f}%] |",
    ]
    if logistic_mean is not None:
        lines.append(f"| logistic_regression | {logistic_mean:.1f}% | — |")
    if xgb_mean is not None:
        lines.append(f"| xgboost_shallow | {xgb_mean:.1f}% | — |")

    lines.extend(
        [
            "",
            "## How to reproduce",
            "",
            "```bash",
            "qml-train --model hybrid_sandwich --dataset breast_cancer --profile publication",
            "# or full benchmark:",
            "python experiments/exp_024_quantum_nano_bc/run.py --profile publication",
            "```",
            "",
            "## Artifacts",
            "",
        ]
    )
    if ckpt:
        lines.append(f"- Best checkpoint: `{ckpt}`")
    else:
        lines.append("- Checkpoint: run with `--profile publication` and `save_checkpoints=true`")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Simulator-only quantum execution (PennyLane `default.qubit`)",
            "- Nano parameter budget (~150–300 trainable parameters)",
            "- Single holdout protocol; no nested cross-validation",
            "- Results vary by seed list in `config/experiments.yaml`",
            "",
            "## Citation",
            "",
            "See [CITATION.cff](../CITATION.cff) and `experiments/exp_024_quantum_nano_bc/hypothesis.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_quantum_nano_bc_card(
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    output_path: Path = DEFAULT_OUT,
) -> Path:
    content = generate_quantum_nano_bc_card(jsonl_path=jsonl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate QuantumNano-BC model card")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = write_quantum_nano_bc_card(jsonl_path=args.jsonl, output_path=args.output)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
