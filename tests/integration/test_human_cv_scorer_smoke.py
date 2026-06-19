"""Integration — score human patient profile through exp_034 checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.human_cv_scorer import PatientProfile, score_patient
from src.shared.result import Ok


@pytest.mark.integration
def test_score_healthy_vs_sick_profiles():
    root = Path(__file__).resolve().parents[2]
    ckpt = root / "artifacts" / "exp_034" / "large_nano_mlp" / "seed_42" / "best.pt"
    if not ckpt.is_file():
        pytest.skip("exp_034 checkpoint missing")

    healthy = PatientProfile(age_years=30, sex_male=False, bmi=22, systolic_bp=110, smoker=False)
    sick = PatientProfile(
        age_years=80, sex_male=True, bmi=35, systolic_bp=170,
        smoker=True, diabetes=True, prior_mi=True,
    )

    healthy_out = score_patient(healthy)
    sick_out = score_patient(sick)
    assert isinstance(healthy_out, Ok) and isinstance(sick_out, Ok)
    assert sick_out.value.probability >= healthy_out.value.probability
