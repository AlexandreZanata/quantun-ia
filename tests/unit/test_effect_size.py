"""Unit tests for paired effect size metrics."""

from src.training.statistics import cohens_d_paired, paired_comparison


def test_cohens_d_paired_positive_mean_diff():
    a = [0.7, 0.72, 0.68, 0.71]
    b = [0.6, 0.62, 0.58, 0.61]
    d = cohens_d_paired(a, b)
    assert d > 0


def test_paired_comparison_includes_effect_size():
    a = [0.7, 0.72, 0.68, 0.71]
    b = [0.6, 0.62, 0.58, 0.61]
    result = paired_comparison(a, b)
    assert "effect_size_cohens_d" in result
    assert result["effect_size_cohens_d"] == cohens_d_paired(a, b)
