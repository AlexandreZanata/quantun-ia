"""Unit tests — literature-backed clinical validation cases."""

from __future__ import annotations

from src.application.clinical_validation_cases import (
    CLINICAL_VALIDATION_CASES,
    case_by_id,
    high_risk_cases,
    low_risk_cases,
)


def test_eight_cases_with_unique_ids():
    assert len(CLINICAL_VALIDATION_CASES) == 8
    ids = [c.case_id for c in CLINICAL_VALIDATION_CASES]
    assert len(set(ids)) == 8
    assert ids[:4] == ["L01", "L02", "L03", "L04"]
    assert ids[4:] == ["H01", "H02", "H03", "H04"]


def test_expected_ranks_monotonic_1_to_8():
    ranks = [c.expected_rank for c in CLINICAL_VALIDATION_CASES]
    assert ranks == list(range(1, 9))


def test_low_and_high_partitions():
    assert len(low_risk_cases()) == 4
    assert len(high_risk_cases()) == 4


def test_case_by_id():
    case = case_by_id("H02")
    assert case.title == "Post-MI diabetic smoker"
    assert case.profile.prior_mi is True
