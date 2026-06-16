"""Research-grade statistical summaries for multi-seed experiments."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))


def seed_summary(values: list[float] | np.ndarray) -> dict:
    """Descriptive stats + bootstrap CI for holdout accuracies across seeds."""
    arr = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_seeds": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def paired_comparison(
    condition_a: list[float],
    condition_b: list[float],
    alpha: float = 0.05,
) -> dict:
    """Paired Wilcoxon signed-rank test (non-parametric, seed-aligned)."""
    a = np.asarray(condition_a, dtype=np.float64)
    b = np.asarray(condition_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("Paired comparison requires equal-length seed results")
    if len(a) < 2:
        return {
            "test": "wilcoxon",
            "statistic": None,
            "p_value": None,
            "significant": None,
            "mean_diff": float(np.mean(a - b)),
            "n_pairs": len(a),
        }

    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": 0.0,
            "n_pairs": len(a),
        }

    stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
    return {
        "test": "wilcoxon",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_diff": float(np.mean(diffs)),
        "n_pairs": len(a),
        "alpha": alpha,
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """
    Holm-Bonferroni step-down correction for multiple paired comparisons.
    Returns list aligned with input order: p_raw, p_adjusted, significant_holm.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1] if item[1] is not None else 1.0)
    adjusted_by_index: dict[int, dict] = {}
    prev_adjusted = 0.0

    for rank, (idx, p_raw) in enumerate(indexed):
        if p_raw is None:
            adjusted_by_index[idx] = {
                "p_raw": None,
                "p_adjusted": None,
                "significant_holm": None,
            }
            continue
        factor = m - rank
        adjusted = min(1.0, max(p_raw * factor, prev_adjusted))
        prev_adjusted = adjusted
        adjusted_by_index[idx] = {
            "p_raw": float(p_raw),
            "p_adjusted": float(adjusted),
            "significant_holm": bool(adjusted < alpha),
        }

    return [adjusted_by_index[i] for i in range(m)]
