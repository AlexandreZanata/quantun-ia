"""Tests for effect size magnitude and MDE helpers."""

import math

from src.training.effect_size import (
    cohens_d_magnitude,
    format_cohens_d,
    minimum_detectable_effect,
)


def test_cohens_d_magnitude_labels():
    assert cohens_d_magnitude(0.1) == "negligible"
    assert cohens_d_magnitude(-0.35) == "small"
    assert cohens_d_magnitude(0.6) == "medium"
    assert cohens_d_magnitude(1.2) == "large"


def test_minimum_detectable_effect_decreases_with_n():
    mde_5 = minimum_detectable_effect(5)
    mde_10 = minimum_detectable_effect(10)
    assert mde_10 < mde_5


def test_minimum_detectable_effect_inf_for_single_pair():
    assert math.isinf(minimum_detectable_effect(1))


def test_format_cohens_d_includes_magnitude():
    text = format_cohens_d(0.75)
    assert "0.75" in text
    assert "medium" in text
