"""Export bootstrap CI summary tables as LaTeX snippets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dashboard.benchmark_data import latest_multi_seed_summaries, load_records

DEFAULT_JSONL = Path("logs/experiments.jsonl")
DEFAULT_OUT = Path("paper/tables")


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def _load_summaries(jsonl_path: Path) -> dict[str, dict[str, dict]]:
    if jsonl_path == DEFAULT_JSONL:
        records = load_records()
    else:
        records: list[dict] = []
        if jsonl_path.exists():
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return latest_multi_seed_summaries(records)


def summary_to_latex(
    exp_id: str,
    summary: dict[str, dict],
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Convert one experiment multi-seed summary to a LaTeX tabular block."""
    cap = caption or f"Holdout accuracy for {exp_id} (mean $\\pm$ 95\\% bootstrap CI)."
    lbl = label or f"tab:{exp_id}"
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{{lbl}}}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Model & Mean (\\%) & 95\\% CI \\\\",
        "\\midrule",
    ]
    for model, stats in sorted(summary.items(), key=lambda kv: kv[1]["mean"], reverse=True):
        mean_pct = stats["mean"] * 100
        ci_low = stats["ci_low"] * 100
        ci_high = stats["ci_high"] * 100
        lines.append(f"{_latex_escape(model)} & {mean_pct:.1f} & [{ci_low:.1f}, {ci_high:.1f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def export_latex_tables(
    jsonl_path: Path = DEFAULT_JSONL,
    out_dir: Path = DEFAULT_OUT,
) -> list[Path]:
    """Write one .tex file per experiment with multi-seed summary."""
    summaries = _load_summaries(jsonl_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for exp_id in sorted(summaries):
        if not exp_id.startswith("exp_"):
            continue
        tex = summary_to_latex(exp_id, summaries[exp_id])
        path = out_dir / f"{exp_id}_holdout.tex"
        path.write_text(tex + "\n", encoding="utf-8")
        created.append(path)

    combined_path = out_dir / "all_experiments_summary.tex"
    if created:
        combined_lines = ["% Auto-generated holdout summary tables", ""]
        for path in created:
            if path.name == "all_experiments_summary.tex":
                continue
            combined_lines.append(f"\\input{{tables/{path.stem}}}")
            combined_lines.append("")
        combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
        created.append(combined_path)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LaTeX tables from experiment logs")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    created = export_latex_tables(jsonl_path=args.jsonl, out_dir=args.out)
    if not created:
        print("No LaTeX tables exported (no multi_seed_summary records).")
        return
    print(f"Exported {len(created)} LaTeX file(s) to {args.out}:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
