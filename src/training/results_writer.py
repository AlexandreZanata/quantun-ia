"""Generate results.md content from experiment JSONL summaries."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from dashboard.benchmark_data import latest_multi_seed_summaries, load_records

DEFAULT_JSONL = Path("logs/experiments.jsonl")


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


def _latest_paired_batch(records: list[dict], exp_id: str) -> list[dict] | None:
    latest_at = ""
    latest_comparisons: list[dict] | None = None
    for record in records:
        if record.get("record_type") != "paired_comparison_batch":
            continue
        if record.get("exp_id") != exp_id:
            continue
        started = record.get("started_at") or ""
        if started >= latest_at:
            latest_at = started
            latest_comparisons = record.get("comparisons", [])
    return latest_comparisons


def _infer_profile(records: list[dict], exp_id: str) -> str:
    for record in reversed(records):
        if record.get("exp_id") == exp_id and record.get("profile"):
            return str(record["profile"])
    return "publication"


def generate_results_md(
    exp_id: str,
    exp_title: str,
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    dataset_note: str = "",
    conclusion_hint: str = "",
) -> str:
    """Build results.md markdown from the latest multi-seed summary in JSONL."""
    records = _load_records(jsonl_path)
    summaries = latest_multi_seed_summaries(records)
    summary = summaries.get(exp_id)
    if not summary:
        raise ValueError(f"No multi_seed_summary found for {exp_id} in {jsonl_path}")

    profile = _infer_profile(records, exp_id)
    n_seeds = next(iter(summary.values())).get("n_seeds", 0)
    lines = [
        f"# Results — {exp_title}",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        f"**Profile:** {profile}, {n_seeds} seeds",
    ]
    if dataset_note:
        lines.append(f"**Dataset:** {dataset_note}")
    lines.append("")
    lines.extend(["## Holdout results", "| Model | Mean | 95% CI |", "|-------|------|--------|"])

    ranked = sorted(summary.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    best_model = ranked[0][0]
    for model, stats in ranked:
        mean_pct = stats["mean"] * 100
        ci_low = stats["ci_low"] * 100
        ci_high = stats["ci_high"] * 100
        marker = "**" if model == best_model else ""
        end = "**" if model == best_model else ""
        lines.append(f"| {marker}{model}{end} | {mean_pct:.1f}% | [{ci_low:.1f}%, {ci_high:.1f}%] |")

    comparisons = _latest_paired_batch(records, exp_id)
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired Wilcoxon (Holm-Bonferroni where batched)",
                "| Comparison | Mean diff | p-value | Cohen's d | Significant |",
                "|------------|-----------|---------|-----------|-------------|",
            ]
        )
        for comp in comparisons:
            label_a = comp.get("label_a", "?")
            label_b = comp.get("label_b", "?")
            mean_diff = comp.get("mean_diff", 0) * 100
            p_val = comp.get("p_value")
            p_holm = comp.get("p_value_holm", p_val)
            d = comp.get("effect_size_cohens_d")
            sig = comp.get("significant_holm", comp.get("significant"))
            sig_txt = "yes" if sig else "no"
            p_txt = f"{p_holm:.3f}" if p_holm is not None else "—"
            d_txt = f"{d:.2f}" if d is not None else "—"
            lines.append(f"| {label_a} vs {label_b} | {mean_diff:+.1f} pp | {p_txt} | {d_txt} | {sig_txt} |")

    lines.extend(["", "## Conclusion"])
    if conclusion_hint:
        lines.append(conclusion_hint)
    else:
        best = ranked[0]
        lines.append(
            f"{best[0]} leads with {best[1]['mean']*100:.1f}% mean holdout "
            f"(95% CI [{best[1]['ci_low']*100:.1f}%, {best[1]['ci_high']*100:.1f}%]). "
            "See paired comparisons for significance."
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- Single holdout split protocol; no nested CV.",
            "- Results specific to the profile and seed list in `config/experiments.yaml`.",
            "- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).",
            "",
        ]
    )
    return "\n".join(lines)


def write_results_md(
    exp_id: str,
    exp_dir: Path,
    *,
    exp_title: str,
    jsonl_path: Path = DEFAULT_JSONL,
    dataset_note: str = "",
    conclusion_hint: str = "",
) -> Path:
    """Write results.md into the experiment folder."""
    content = generate_results_md(
        exp_id,
        exp_title,
        jsonl_path=jsonl_path,
        dataset_note=dataset_note,
        conclusion_hint=conclusion_hint,
    )
    out = exp_dir / "results.md"
    out.write_text(content, encoding="utf-8")
    return out
