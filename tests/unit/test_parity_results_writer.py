"""Unit tests for exp_022 parity results.md writer."""

from __future__ import annotations

from src.application.dto import NanoParityBenchResult
from src.application.parity_results_writer import generate_parity_results_md


def _sample_outcome(*, verdict: str = "inconclusive", quantum_wins: bool = False) -> NanoParityBenchResult:
    return NanoParityBenchResult(
        exp_id="exp_022",
        quantum_model="hybrid_sandwich",
        dataset="wine_binary",
        profile="publication",
        classical_label="classical_matched_h6",
        quantum_n_params=85,
        classical_n_params=91,
        classical_hidden=6,
        param_delta=-6,
        quantum_accuracies=[0.95, 0.94],
        classical_accuracies=[0.92, 0.91],
        quantum_mean=0.945,
        classical_mean=0.915,
        quantum_summary={"mean": 0.945, "std": 0.01, "ci_low": 0.93, "ci_high": 0.96, "n_seeds": 2},
        classical_summary={"mean": 0.915, "std": 0.01, "ci_low": 0.90, "ci_high": 0.93, "n_seeds": 2},
        comparison={
            "label_a": "hybrid_sandwich",
            "label_b": "classical_matched_h6",
            "mean_diff": 0.03,
            "p_value": 0.25,
            "significant": False,
            "effect_size_cohens_d": 0.5,
            "effect_size_magnitude": "medium",
        },
        quantum_wins=quantum_wins,
        verdict=verdict,
        datasets_status={"wine_binary": "ready"},
    )


def test_generate_parity_results_md_has_required_sections():
    text = generate_parity_results_md([_sample_outcome()])
    for section in (
        "## Holdout results",
        "## Paired Wilcoxon",
        "## Verdict",
        "## Power analysis",
        "## Conclusion",
        "## Limitations",
    ):
        assert section in text


def test_generate_parity_results_md_inconclusive_verdict():
    text = generate_parity_results_md([_sample_outcome(verdict="inconclusive")])
    assert "**inconclusive**" in text
