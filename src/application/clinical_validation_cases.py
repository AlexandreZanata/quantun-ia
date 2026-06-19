"""Fixed clinical validation cases — literature-backed profiles for human-facing model tests."""

from __future__ import annotations

from dataclasses import dataclass

from src.application.human_cv_scorer import PatientProfile

# Expected rank 1 = lowest clinical risk, 8 = highest (see exp_041 hypothesis.md)


@dataclass(frozen=True)
class ClinicalValidationCase:
    case_id: str
    title: str
    expected_tier: str
    expected_rank: int
    science_note: str
    profile: PatientProfile


CLINICAL_VALIDATION_CASES: tuple[ClinicalValidationCase, ...] = (
    ClinicalValidationCase(
        case_id="L01",
        title="Young healthy woman",
        expected_tier="very_low",
        expected_rank=1,
        science_note=(
            "Young female, optimal BMI/BP/lipids, non-smoker — pooled-cohort 10y ASCVD risk typically <5% "
            "(Arnett et al., 2019)."
        ),
        profile=PatientProfile(
            age_years=28,
            sex_male=False,
            bmi=21.5,
            systolic_bp=108,
            diastolic_bp=68,
            heart_rate=68,
            total_cholesterol=175,
            hdl=68,
            triglycerides=90,
            glucose=88,
            hba1c=5.2,
            smoker=False,
            diabetes=False,
            family_history_cvd=False,
        ),
    ),
    ClinicalValidationCase(
        case_id="L02",
        title="Active man in his 30s",
        expected_tier="very_low",
        expected_rank=2,
        science_note=(
            "Age 34, normal BMI and BP, favorable HDL — low short-term ASCVD event rate in primary prevention cohorts."
        ),
        profile=PatientProfile(
            age_years=34,
            sex_male=True,
            bmi=23.5,
            systolic_bp=115,
            diastolic_bp=72,
            heart_rate=62,
            total_cholesterol=185,
            hdl=58,
            triglycerides=100,
            glucose=92,
            hba1c=5.3,
            smoker=False,
            diabetes=False,
        ),
    ),
    ClinicalValidationCase(
        case_id="L03",
        title="Middle-aged woman, borderline factors",
        expected_tier="low",
        expected_rank=3,
        science_note=(
            "Age 48, mild overweight, BP near 130/80 — below treatment threshold but above optimal (ACC/AHA 2017)."
        ),
        profile=PatientProfile(
            age_years=48,
            sex_male=False,
            bmi=25.5,
            systolic_bp=128,
            diastolic_bp=82,
            total_cholesterol=195,
            hdl=52,
            triglycerides=130,
            glucose=98,
            hba1c=5.6,
            smoker=False,
            diabetes=False,
        ),
    ),
    ClinicalValidationCase(
        case_id="L04",
        title="50-year-old man, mild hypertension",
        expected_tier="low",
        expected_rank=4,
        science_note=(
            "Stage-1 hypertension range (130–139 systolic), overweight — intermediate lifetime risk without ASCVD history."
        ),
        profile=PatientProfile(
            age_years=50,
            sex_male=True,
            bmi=26.5,
            systolic_bp=132,
            diastolic_bp=84,
            total_cholesterol=210,
            hdl=48,
            triglycerides=145,
            glucose=102,
            hba1c=5.7,
            smoker=False,
            diabetes=False,
        ),
    ),
    ClinicalValidationCase(
        case_id="H01",
        title="Obese smoker with diabetes and HTN",
        expected_tier="high",
        expected_rank=5,
        science_note=(
            "Smoking + diabetes + obesity + hypertension — multiplicative risk per Framingham and ACC/AHA charts."
        ),
        profile=PatientProfile(
            age_years=58,
            sex_male=True,
            bmi=34,
            systolic_bp=158,
            diastolic_bp=92,
            total_cholesterol=265,
            hdl=36,
            triglycerides=210,
            glucose=145,
            hba1c=7.8,
            smoker=True,
            diabetes=True,
            family_history_cvd=True,
        ),
    ),
    ClinicalValidationCase(
        case_id="H02",
        title="Post-MI diabetic smoker",
        expected_tier="high",
        expected_rank=6,
        science_note=(
            "Secondary-prevention profile: prior MI and diabetes — high recurrent event risk (secondary prevention guidelines)."
        ),
        profile=PatientProfile(
            age_years=67,
            sex_male=True,
            bmi=29,
            systolic_bp=152,
            diastolic_bp=88,
            total_cholesterol=240,
            hdl=38,
            glucose=138,
            hba1c=7.5,
            smoker=True,
            diabetes=True,
            prior_mi=True,
            statin=True,
            beta_blocker=True,
            aspirin=True,
        ),
    ),
    ClinicalValidationCase(
        case_id="H03",
        title="Elderly woman, stroke and atrial fibrillation",
        expected_tier="very_high",
        expected_rank=7,
        science_note=(
            "Prior stroke + AF + diabetes in late 70s — very high 1y recurrence/stroke risk (CHADS-VASc context)."
        ),
        profile=PatientProfile(
            age_years=76,
            sex_male=False,
            bmi=28,
            systolic_bp=162,
            diastolic_bp=92,
            total_cholesterol=220,
            hdl=42,
            glucose=128,
            hba1c=7.2,
            diabetes=True,
            prior_stroke=True,
            atrial_fibrillation=True,
            ace_inhibitor=True,
        ),
    ),
    ClinicalValidationCase(
        case_id="H04",
        title="Multimorbid elderly man",
        expected_tier="very_high",
        expected_rank=8,
        science_note=(
            "Age 81 with prior MI, stroke, COPD, CKD, smoker — maximal epidemiologic risk cluster."
        ),
        profile=PatientProfile(
            age_years=81,
            sex_male=True,
            bmi=27,
            systolic_bp=168,
            diastolic_bp=94,
            total_cholesterol=230,
            hdl=40,
            glucose=142,
            hba1c=7.6,
            smoker=True,
            diabetes=True,
            prior_mi=True,
            prior_stroke=True,
            copd=True,
            ckd_stage=2,
            atrial_fibrillation=True,
            statin=True,
            aspirin=True,
        ),
    ),
)


def low_risk_cases() -> list[ClinicalValidationCase]:
    return [c for c in CLINICAL_VALIDATION_CASES if c.case_id.startswith("L")]


def high_risk_cases() -> list[ClinicalValidationCase]:
    return [c for c in CLINICAL_VALIDATION_CASES if c.case_id.startswith("H")]


def case_by_id(case_id: str) -> ClinicalValidationCase:
    for case in CLINICAL_VALIDATION_CASES:
        if case.case_id == case_id:
            return case
    msg = f"unknown case_id: {case_id}"
    raise KeyError(msg)
