"""Generate publication-quality figures from logs/experiments.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dashboard.benchmark_data import (
    latest_holdout_records,
    load_records,
    to_leaderboard_rows,
)
from src.training.plot_style import apply_publication_style

DEFAULT_JSONL = Path("logs/experiments.jsonl")
DEFAULT_OUT = Path("figures")


def _load_records(jsonl_path: Path) -> list[dict]:
    if jsonl_path == DEFAULT_JSONL:
        return load_records()
    records: list[dict] = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def plot_experiment_leaderboard(
    exp_id: str,
    rows: list[dict],
    out_path: Path,
) -> bool:
    """Bar chart with bootstrap CI error bars for one experiment."""
    exp_rows = [r for r in rows if r["exp_id"] == exp_id and r.get("accuracy") is not None]
    if not exp_rows:
        return False

    exp_rows = sorted(exp_rows, key=lambda r: r["accuracy"], reverse=True)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    models = [r["model"] for r in exp_rows]
    accs = [r["accuracy"] for r in exp_rows]
    yerr_low = [r["accuracy"] - r["ci_low"] if r.get("ci_low") is not None else 0 for r in exp_rows]
    yerr_high = [r["ci_high"] - r["accuracy"] if r.get("ci_high") is not None else 0 for r in exp_rows]
    yerr = np.array([yerr_low, yerr_high])

    x = np.arange(len(models))
    ax.bar(x, accs, yerr=yerr if any(yerr_low + yerr_high) else None, capsize=3, color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Holdout accuracy (%)")
    ax.set_ylim(0, min(100, max(accs) + 15))
    ax.set_title(f"{exp_id} — holdout leaderboard")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return True


def plot_cross_experiment_best(rows: list[dict], out_path: Path) -> bool:
    """Best model per experiment (highest mean accuracy)."""
    by_exp: dict[str, dict] = {}
    for row in rows:
        if row.get("accuracy") is None:
            continue
        exp_id = row["exp_id"]
        prev = by_exp.get(exp_id)
        if prev is None or row["accuracy"] > prev["accuracy"]:
            by_exp[exp_id] = row

    if not by_exp:
        return False

    ordered = sorted(by_exp.items(), key=lambda kv: kv[0])
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.5))

    labels = [f"{exp}\n{best['model']}" for exp, best in ordered]
    accs = [best["accuracy"] for _, best in ordered]
    x = np.arange(len(labels))
    ax.bar(x, accs, color="#009E73")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best holdout accuracy (%)")
    ax.set_title("Best model per experiment")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return True


def plot_learning_curves(records: list[dict], out_path: Path, max_traces: int = 12) -> bool:
    """Overlay training accuracy curves from holdout records with history."""
    traces: list[tuple[str, list[int], list[float]]] = []
    for record in latest_holdout_records(records):
        history = record.get("history") or []
        if not history or "epoch" not in history[0]:
            continue
        label = f"{record.get('exp_id', '?')}:{record.get('model_name', '?')}"
        epochs = [h["epoch"] for h in history]
        accs = [h.get("accuracy", 0) * 100 for h in history]
        traces.append((label, epochs, accs))

    if not traces:
        return False

    traces = traces[:max_traces]
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, epochs, accs in traces:
        ax.plot(epochs, accs, label=label, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy (%)")
    ax.set_title("Learning curves (latest holdout runs)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return True


def generate_all_figures(
    jsonl_path: Path = DEFAULT_JSONL,
    out_dir: Path = DEFAULT_OUT,
) -> list[Path]:
    """Generate all publication figures. Returns paths of created files."""
    records = _load_records(jsonl_path)
    rows = to_leaderboard_rows(records)
    created: list[Path] = []

    exp_ids = sorted({r["exp_id"] for r in rows})
    for exp_id in exp_ids:
        out = out_dir / f"{exp_id}_leaderboard.pdf"
        if plot_experiment_leaderboard(exp_id, rows, out):
            created.append(out)

    summary = out_dir / "cross_experiment_best.pdf"
    if plot_cross_experiment_best(rows, summary):
        created.append(summary)

    curves = out_dir / "learning_curves.pdf"
    if plot_learning_curves(records, curves):
        created.append(curves)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from experiment logs")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Path to experiments.jsonl")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for PDFs")
    args = parser.parse_args()

    created = generate_all_figures(jsonl_path=args.jsonl, out_dir=args.out)
    if not created:
        print("No figures generated (empty or missing logs).")
        return
    print(f"Generated {len(created)} figure(s) in {args.out}:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
