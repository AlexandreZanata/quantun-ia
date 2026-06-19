"""Unit tests — human CV scorer profile ↔ feature mapping."""

from __future__ import annotations

import pytest

from src.application.human_cv_scorer import (
    PatientProfile,
    features_to_profile,
    profile_summary,
    profile_to_features,
    risk_band,
)


def test_profile_to_features_length():
    vector = profile_to_features(PatientProfile())
    assert len(vector) == 40


def test_hypertension_flag_from_systolic_bp():
    low_bp = profile_to_features(PatientProfile(systolic_bp=120.0))
    high_bp = profile_to_features(PatientProfile(systolic_bp=150.0))
    assert low_bp[15] == 0.0
    assert high_bp[15] == 1.0


def test_features_to_profile_roundtrip_key_fields():
    original = PatientProfile(
        age_years=62.0,
        sex_male=True,
        bmi=31.2,
        systolic_bp=142.0,
        smoker=True,
        diabetes=True,
    )
    restored = features_to_profile(profile_to_features(original))
    assert restored.age_years == pytest.approx(62.0)
    assert restored.sex_male is True
    assert restored.bmi == pytest.approx(31.2)
    assert restored.systolic_bp == pytest.approx(142.0)
    assert restored.smoker is True
    assert restored.diabetes is True


def test_risk_band_thresholds():
    assert risk_band(0.2)[0] == "low"
    assert risk_band(0.55)[0] == "moderate"
    assert risk_band(0.85)[0] == "high"


def test_profile_summary_readable():
    text = profile_summary(PatientProfile(age_years=45, sex_male=False, smoker=True))
    assert "45y female" in text
    assert "smoker" in text
