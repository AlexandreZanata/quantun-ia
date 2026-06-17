"""Export append-only JSONL experiment logs to CSV for DVC and publication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

DEFAULT_JSONL = Path("logs/experiments.jsonl")
DEFAULT_CSV = Path("data/exports/results.csv")

SUMMARY_FIELDS = [
    "exp_id",
    "model_name",
    "record_type",
    "seed",
    "profile",
    "started_at",
    "elapsed_s",
    "final_acc",
    "test_accuracy",
    "test_loss",
    "eval_set",
    "n_epochs",
    "n_params",
]


def _flatten_record(record: dict) -> dict:
    row = {field: record.get(field) for field in SUMMARY_FIELDS}
    row["record_type"] = record.get("record_type", "experiment")
    return row


def export_jsonl_to_csv(
    jsonl_path: Path = DEFAULT_JSONL,
    csv_path: Path = DEFAULT_CSV,
) -> int:
    """Write experiment summary rows to CSV. Returns number of rows exported."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type") in {
                    "multi_seed_summary",
                    "paired_comparison",
                    "paired_comparison_batch",
                    "applicability_gate",
                }:
                    rows.append(
                        {
                            "exp_id": record.get("exp_id"),
                            "model_name": record.get("model_name"),
                            "record_type": record.get("record_type"),
                            "started_at": record.get("started_at"),
                        }
                    )
                else:
                    rows.append(_flatten_record(record))

    fieldnames = list(SUMMARY_FIELDS)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    count = export_jsonl_to_csv()
    print(f"Exported {count} rows to {DEFAULT_CSV}")
