"""Unit tests for human agro scorer feature mapping."""

from src.application.agro_validation_cases import AGRO_VALIDATION_CASES
from src.application.human_agro_scorer import (
    profile_to_features,
    top_climate_drivers,
    yield_risk_tier,
)


def test_profile_to_features_shape():
    from src.data.open_acyd import N_FEATURES as ACYD_N

    profile = AGRO_VALIDATION_CASES[0].profile
    vector = profile_to_features(profile)
    assert len(vector) == ACYD_N == 37


def test_yield_risk_tier_bands():
    assert yield_risk_tier(0.2)[0] == "low"
    assert yield_risk_tier(0.5)[0] == "moderate"
    assert yield_risk_tier(0.8)[0] == "high"


def test_top_climate_drivers_returns_three():
    profile = AGRO_VALIDATION_CASES[-1].profile
    drivers = top_climate_drivers(profile)
    assert len(drivers) == 3
