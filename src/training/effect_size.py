"""Effect size interpretation and power analysis helpers."""

from __future__ import annotations

from scipy import stats


def cohens_d_magnitude(d: float) -> str:
    """Map |Cohen's d| to conventional magnitude label."""
    if d != d:  # NaN
        return "n/a"
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def minimum_detectable_effect(
    n_pairs: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Approximate minimum paired Cohen's d detectable at given alpha and power.

    Uses normal approximation for paired t-test sample size (two-sided).
    """
    if n_pairs < 2:
        return float("inf")
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))
    return (z_alpha + z_beta) / (n_pairs**0.5)


def format_cohens_d(d: float | None) -> str:
    """Format Cohen's d with magnitude label for results.md tables."""
    if d is None or d != d:
        return "—"
    return f"{d:.2f} ({cohens_d_magnitude(d)})"
