"""Unit tests for data provenance and profile validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.data_provenance_check import (
    check_dataset_provenance,
    validate_patient_profile,
)
from src.application.human_cv_scorer import PatientProfile


def test_validate_patient_profile_plausible():
    result = validate_patient_profile(PatientProfile())
    assert result.is_plausible
    assert result.trust_level == "plausible"
    assert result.checks_passed == result.checks_total


def test_validate_patient_profile_implausible_bp():
    profile = PatientProfile(systolic_bp=60.0, diastolic_bp=90.0)
    result = validate_patient_profile(profile)
    assert not result.is_plausible
    assert result.trust_level == "implausible"
    assert any("Systolic" in w for w in result.warnings)


def test_validate_patient_profile_young_mi_warning():
    profile = PatientProfile(age_years=22.0, prior_mi=True)
    result = validate_patient_profile(profile)
    assert result.trust_level == "review"
    assert any("Prior MI" in w for w in result.warnings)


@pytest.mark.skipif(
    not Path("data/open/manifest.json").is_file(),
    reason="open data manifest not present",
)
def test_check_dataset_provenance_synthea():
    result = check_dataset_provenance("synthea_cv_risk_v1")
    assert result.is_synthetic
    assert result.badge_label == "SYNTHETIC DATA"
    assert result.origin == "synthetic"
