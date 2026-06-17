"""MicroQML Bench v1 — versioned export bundle for external citation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dashboard.benchmark_data import load_records, to_leaderboard_rows

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "microqml_bench" / "v1.yaml"
DEFAULT_EXPORT_DIR = ROOT / "dist" / "microqml_bench"


def load_bench_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load a MicroQML Bench YAML definition."""
    path = config_path or DEFAULT_CONFIG_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid bench config: {path}")
    return data


def _task_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["exp_id"]: task for task in config.get("tasks", [])}


def _leaderboard_entry(task: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "exp_id": row["exp_id"],
        "model": row["model"],
        "accuracy_pct": row["accuracy"],
        "ci_low_pct": row.get("ci_low"),
        "ci_high_pct": row.get("ci_high"),
        "std_pct": row.get("std_pct"),
        "n_seeds": row.get("n_seeds"),
        "n_params": row.get("n_params"),
        "elapsed_s": row.get("elapsed_s"),
        "epochs": row.get("epochs"),
        "eval_set": row.get("eval_set", "holdout_test"),
        "source": row.get("source", "unknown"),
    }


def build_bench_export(
    records: list[dict] | None = None,
    *,
    config_path: Path | None = None,
    software_version: str | None = None,
) -> dict[str, Any]:
    """Build a MicroQML Bench v1 JSON document from experiment logs."""
    config = load_bench_config(config_path)
    if records is None:
        records = load_records()

    tasks_by_exp = _task_index(config)
    leaderboard: list[dict[str, Any]] = []
    for row in to_leaderboard_rows(records):
        task = tasks_by_exp.get(row["exp_id"])
        if task is None:
            continue
        if row.get("accuracy") is None:
            continue
        leaderboard.append(_leaderboard_entry(task, row))

    return {
        "bench_id": config["bench_id"],
        "version": config["version"],
        "schema_version": config["schema_version"],
        "display_name": config["display_name"],
        "description": config.get("description", "").strip(),
        "protocol": config["protocol"],
        "tasks": config["tasks"],
        "leaderboard": leaderboard,
        "generated_at": datetime.now(UTC).isoformat(),
        "software_version": software_version or _read_software_version(),
        "citation": config.get("citation", "").strip(),
    }


def _read_software_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    return "unknown"


def write_bench_export(
    export: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Write export JSON to disk (default dist/microqml_bench/v1.json)."""
    path = output_path or (DEFAULT_EXPORT_DIR / "v1.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return path
