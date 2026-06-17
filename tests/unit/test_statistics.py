"""Unit tests for research statistics module."""


from src.training.statistics import bootstrap_ci, paired_comparison, seed_summary


def test_bootstrap_ci_single_value():
    low, high = bootstrap_ci([0.85])
    assert low == high == 0.85


def test_seed_summary_includes_ci():
    values = [0.80, 0.85, 0.90]
    stats = seed_summary(values)

    assert stats["n_seeds"] == 3
    assert stats["mean"] == 0.85
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]


def test_paired_comparison_identical():
    result = paired_comparison([0.8, 0.85, 0.9], [0.8, 0.85, 0.9])
    assert result["mean_diff"] == 0.0
    assert result["significant"] is False


def test_paired_comparison_detects_difference():
    a = [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.96, 0.95]
    b = [0.45, 0.44, 0.46, 0.43, 0.45, 0.44, 0.46, 0.45]
    result = paired_comparison(a, b)
    assert result["mean_diff"] > 0.4
    assert result["p_value"] is not None
    assert result["p_value"] < 0.01
