"""Generate results.md for Nano Parity Bench (exp_022) from bench outcomes."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.application.dto import NanoParityBenchResult
from src.training.effect_size import format_cohens_d, minimum_detectable_effect


def generate_parity_results_md(
    outcomes: list[NanoParityBenchResult],
    *,
    exp_title: str = "EXP 022 (Nano Quantum Parity)",
    conclusion_hint: str = "",
) -> str:
    """Build uniform results.md for hybrid_sandwich vs matched classical parity runs."""
    if not outcomes:
        raise ValueError("at least one NanoParityBenchResult required")

    profile = outcomes[0].profile
    n_seeds = len(outcomes[0].quantum_accuracies)
    lines = [
        f"# Results — {exp_title}",
        "",
        f"**Run date:** {date.today().isoformat()}  ",
        f"**Profile:** {profile}, {n_seeds} seeds",
        "**Protocol:** parameter-matched classical MLP vs quantum nanomodel (|Δparams| ≤ 10)",
        "",
        "## Holdout results",
        "| Dataset | Quantum model | Classical baseline | Quantum mean | Classical mean | Δ (pp) |",
        "|---------|---------------|-------------------|--------------|----------------|--------|",
    ]

    comparisons: list[dict] = []
    for outcome in outcomes:
        q_pct = outcome.quantum_mean * 100
        c_pct = outcome.classical_mean * 100
        diff_pp = outcome.comparison.get("mean_diff", 0.0) * 100
        lines.append(
            f"| {outcome.dataset} | {outcome.quantum_model} | {outcome.classical_label} | "
            f"{q_pct:.1f}% | {c_pct:.1f}% | {diff_pp:+.1f} |"
        )
        comparisons.append(outcome.comparison)

    lines.extend(
        [
            "",
            "## Paired Wilcoxon (Holm-Bonferroni where batched)",
            "| Comparison | Mean diff | p-value | Cohen's d | Significant |",
            "|------------|-----------|---------|-----------|-------------|",
        ]
    )
    for outcome in outcomes:
        comp = outcome.comparison
        label_a = comp.get("label_a", outcome.quantum_model)
        label_b = comp.get("label_b", outcome.classical_label)
        mean_diff = comp.get("mean_diff", 0.0) * 100
        p_val = comp.get("p_value_holm", comp.get("p_value"))
        p_txt = f"{p_val:.3f}" if p_val is not None else "—"
        d_txt = format_cohens_d(comp.get("effect_size_cohens_d"))
        sig = comp.get("significant_holm", comp.get("significant"))
        sig_txt = "yes" if sig else "no"
        lines.append(
            f"| {label_a} vs {label_b} ({outcome.dataset}) | {mean_diff:+.1f} pp | "
            f"{p_txt} | {d_txt} | {sig_txt} |"
        )

    verdicts = [o.verdict for o in outcomes]
    wins = sum(1 for o in outcomes if o.quantum_wins)
    if wins == len(outcomes):
        verdict_line = (
            f"**accepted** — quantum nanomodel significantly outperforms matched classical "
            f"on all {len(outcomes)} primary datasets (≥2 pp, Holm-significant)."
        )
    elif wins > 0:
        verdict_line = (
            f"**inconclusive** — quantum wins on {wins}/{len(outcomes)} datasets; "
            f"not all comparisons Holm-significant at α=0.05."
        )
    elif all(v == "inconclusive" for v in verdicts):
        verdict_line = (
            "**inconclusive** — positive mean gaps on some datasets but Wilcoxon not "
            "Holm-significant at α=0.05 (underpowered or high variance)."
        )
    else:
        verdict_line = (
            "**rejected** — hybrid_sandwich does not beat parameter-matched classical "
            "at equal param budget on primary datasets."
        )

    mde = minimum_detectable_effect(n_seeds)
    lines.extend(
        [
            "",
            "## Verdict",
            verdict_line,
            "",
            "## Power analysis",
            f"- Design: {n_seeds} paired holdout accuracies per dataset (profile `{profile}`).",
            f"- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **{mde:.2f}**.",
            "- Primary claim threshold: ≥2 pp mean difference + Holm-significant Wilcoxon.",
            "",
            "## Conclusion",
        ]
    )
    if conclusion_hint:
        lines.append(conclusion_hint)
    else:
        lines.append(
            "Hybrid sandwich quantum nanomodel vs parameter-matched classical MLP on UCI tabular. "
            "See `qml-bench-parity` and `config/nano_parity_bench.yaml`."
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "- Single holdout split per seed; no nested CV.",
            "- Classical baseline hidden size chosen by parameter count, not architecture search.",
            "- Results specific to `nano_parity_bench.yaml` seeds and epochs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_parity_results_md(
    outcomes: list[NanoParityBenchResult],
    exp_dir: Path,
    **kwargs,
) -> Path:
    """Write results.md into the experiment folder."""
    content = generate_parity_results_md(outcomes, **kwargs)
    out = exp_dir / "results.md"
    out.write_text(content, encoding="utf-8")
    return out
