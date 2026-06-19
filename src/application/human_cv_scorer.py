"""Human-readable cardiovascular risk scoring — maps patient forms to model features."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.application.dto import PredictNanomodelDTO
from src.application.predict_nanomodel import execute as predict_execute
from src.data.synthea_cv_risk import FEATURE_SEMANTICS, N_FEATURES
from src.shared.result import Fail, Ok, Result, fail, ok

DEFAULT_EXP_ID = "exp_034"
DEFAULT_MODEL = "large_nano_mlp"
DEFAULT_DATASET = "synthea_cv_risk_v1"
DEFAULT_SEED = 42

RESEARCH_DISCLAIMER = (
    "Research prototype on synthetic data — not for clinical decisions."
)


@dataclass(frozen=True)
class PatientProfile:
    """Fields a human can understand — mapped to 40 model features."""

    age_years: float = 55.0
    sex_male: bool = False
    bmi: float = 28.0
    systolic_bp: float = 128.0
    diastolic_bp: float = 78.0
    heart_rate: float = 74.0
    total_cholesterol: float = 200.0
    hdl: float = 50.0
    triglycerides: float = 150.0
    glucose: float = 105.0
    hba1c: float = 5.8
    creatinine: float = 1.0
    smoker: bool = False
    diabetes: bool = False
    atrial_fibrillation: bool = False
    prior_mi: bool = False
    prior_stroke: bool = False
    copd: bool = False
    ckd_stage: int = 0
    family_history_cvd: bool = False
    statin: bool = False
    ace_inhibitor: bool = False
    beta_blocker: bool = False
    aspirin: bool = False
    ed_visits_12m: int = 0
    inpatient_12m: int = 0
    outpatient_12m: int = 4
    medication_count: int = 3
    latent_risk_factor: float = 0.0


@dataclass(frozen=True)
class HumanCvScoreError:
    code: str
    message: str


@dataclass(frozen=True)
class HumanCvScoreResult:
    probability: float
    risk_percent: float
    risk_band: str
    risk_label: int
    human_summary: str
    feature_vector: list[float]
    checkpoint_path: str
    profile: PatientProfile


def risk_band(probability: float) -> tuple[str, str]:
    """Return (band_code, human_label) from model probability."""
    pct = probability * 100.0
    if pct < 40.0:
        return "low", "Low risk"
    if pct < 70.0:
        return "moderate", "Moderate risk"
    return "high", "High risk"


def profile_to_features(profile: PatientProfile) -> list[float]:
    """Convert human patient profile to raw 40-dim feature vector (Synthea v1 schema)."""
    age_norm = profile.age_years / 100.0
    sex = 1.0 if profile.sex_male else 0.0
    sbp = float(profile.systolic_bp)
    dbp = float(profile.diastolic_bp)
    bmi = float(profile.bmi)
    hr = float(profile.heart_rate)
    chol = float(profile.total_cholesterol)
    hdl = float(profile.hdl)
    ldl = max(50.0, chol - hdl * 0.4)
    tg = float(profile.triglycerides)
    glucose = float(profile.glucose)
    hba1c = float(profile.hba1c)
    creatinine = float(profile.creatinine)
    smoker = 1.0 if profile.smoker else 0.0
    diabetes = 1.0 if profile.diabetes else 0.0
    hypertension = 1.0 if sbp >= 140.0 else 0.0
    afib = 1.0 if profile.atrial_fibrillation else 0.0
    prior_mi = 1.0 if profile.prior_mi else 0.0
    prior_stroke = 1.0 if profile.prior_stroke else 0.0
    copd = 1.0 if profile.copd else 0.0
    ckd = float(max(0, min(3, profile.ckd_stage)))
    statin = 1.0 if profile.statin else 0.0
    ace = 1.0 if profile.ace_inhibitor else 0.0
    beta = 1.0 if profile.beta_blocker else 0.0
    aspirin = 1.0 if profile.aspirin else 0.0
    ed_visits = float(profile.ed_visits_12m)
    inpatient = float(profile.inpatient_12m)
    outpatient = float(profile.outpatient_12m)
    med_count = float(profile.medication_count)
    family_hx = 1.0 if profile.family_history_cvd else 0.0
    latent = float(profile.latent_risk_factor)

    framingham = 0.03 * profile.age_years + 0.8 * sex + 0.04 * sbp + 0.02 * chol - 0.03 * hdl + 0.5 * smoker
    ascvd = 0.025 * profile.age_years + 0.6 * diabetes + 0.4 * hypertension + 0.7 * prior_mi
    charlson = diabetes + copd + ckd + prior_mi * 2 + prior_stroke * 1.5 + afib
    utilization = ed_visits + inpatient * 3 + outpatient * 0.1
    lab_risk = (glucose - 100) / 40 + (hba1c - 5.7) / 1.5 + (creatinine - 1.0) / 0.5
    med_burden = statin + ace + beta + aspirin + med_count * 0.1
    comorbidity = diabetes + hypertension + copd + afib + prior_mi + prior_stroke
    vitals_risk = (sbp - 120) / 20 + (bmi - 25) / 10 + (hr - 70) / 20
    lifestyle = smoker + (1.0 if bmi > 30 else 0.0) + family_hx * 0.5

    vector = [
        age_norm,
        sex,
        bmi,
        sbp,
        dbp,
        hr,
        chol,
        hdl,
        ldl,
        tg,
        glucose,
        hba1c,
        creatinine,
        smoker,
        diabetes,
        hypertension,
        afib,
        prior_mi,
        prior_stroke,
        copd,
        ckd,
        statin,
        ace,
        beta,
        aspirin,
        ed_visits,
        inpatient,
        outpatient,
        med_count,
        family_hx,
        framingham,
        ascvd,
        charlson,
        utilization,
        lab_risk,
        med_burden,
        comorbidity,
        vitals_risk,
        lifestyle,
        latent,
    ]
    if len(vector) != N_FEATURES:
        msg = f"expected {N_FEATURES} features, built {len(vector)}"
        raise ValueError(msg)
    return [float(v) for v in vector]


def features_to_profile(features: list[float]) -> PatientProfile:
    """Reverse-map a raw feature row to a human patient profile (for dataset demos)."""
    if len(features) != N_FEATURES:
        msg = f"expected {N_FEATURES} features, got {len(features)}"
        raise ValueError(msg)
    f = features
    return PatientProfile(
        age_years=round(f[0] * 100.0, 1),
        sex_male=f[1] > 0.5,
        bmi=round(f[2], 1),
        systolic_bp=round(f[3], 0),
        diastolic_bp=round(f[4], 0),
        heart_rate=round(f[5], 0),
        total_cholesterol=round(f[6], 0),
        hdl=round(f[7], 0),
        triglycerides=round(f[9], 0),
        glucose=round(f[10], 0),
        hba1c=round(f[11], 1),
        creatinine=round(f[12], 2),
        smoker=f[13] > 0.5,
        diabetes=f[14] > 0.5,
        atrial_fibrillation=f[16] > 0.5,
        prior_mi=f[17] > 0.5,
        prior_stroke=f[18] > 0.5,
        copd=f[19] > 0.5,
        ckd_stage=int(round(f[20])),
        statin=f[21] > 0.5,
        ace_inhibitor=f[22] > 0.5,
        beta_blocker=f[23] > 0.5,
        aspirin=f[24] > 0.5,
        ed_visits_12m=int(round(f[25])),
        inpatient_12m=int(round(f[26])),
        outpatient_12m=int(round(f[27])),
        medication_count=int(round(f[28])),
        family_history_cvd=f[29] > 0.5,
        latent_risk_factor=f[39],
    )


def profile_summary(profile: PatientProfile) -> str:
    """One-line human-readable patient description."""
    sex = "male" if profile.sex_male else "female"
    flags = []
    if profile.smoker:
        flags.append("smoker")
    if profile.diabetes:
        flags.append("diabetes")
    if profile.prior_mi:
        flags.append("prior heart attack")
    if profile.prior_stroke:
        flags.append("prior stroke")
    flag_text = f", {', '.join(flags)}" if flags else ""
    return (
        f"{int(profile.age_years)}y {sex}, BMI {profile.bmi:.1f}, "
        f"BP {int(profile.systolic_bp)}/{int(profile.diastolic_bp)} mmHg{flag_text}"
    )


def format_risk_message(result: HumanCvScoreResult) -> str:
    """Plain-language outcome for humans."""
    return (
        f"12-month cardiovascular event risk: **{result.risk_percent:.1f}%** "
        f"({result.risk_band.replace('_', ' ').title()})\n\n"
        f"Patient: {result.human_summary}\n\n"
        f"{RESEARCH_DISCLAIMER}"
    )


def score_patient(
    profile: PatientProfile,
    *,
    exp_id: str = DEFAULT_EXP_ID,
    model_name: str = DEFAULT_MODEL,
    dataset: str = DEFAULT_DATASET,
    seed: int = DEFAULT_SEED,
) -> Result[HumanCvScoreResult, HumanCvScoreError]:
    """Score one patient profile through the trained LargeNanoMLP serve checkpoint."""
    features = profile_to_features(profile)
    outcome = predict_execute(
        PredictNanomodelDTO(
            exp_id=exp_id,
            model_name=model_name,
            dataset=dataset,
            seed=seed,
            features=[features],
        )
    )
    if isinstance(outcome, Fail):
        return fail(HumanCvScoreError(outcome.error.code, outcome.error.message))

    assert isinstance(outcome, Ok)
    pred = outcome.value
    prob = pred.probabilities[0]
    band_code, band_label = risk_band(prob)
    return ok(
        HumanCvScoreResult(
            probability=prob,
            risk_percent=prob * 100.0,
            risk_band=band_code,
            risk_label=band_label,
            human_summary=profile_summary(profile),
            feature_vector=features,
            checkpoint_path=pred.checkpoint_path,
            profile=profile,
        )
    )


def compare_profiles(
    profile_a: PatientProfile,
    profile_b: PatientProfile,
) -> Result[tuple[HumanCvScoreResult, HumanCvScoreResult], HumanCvScoreError]:
    """Score two profiles — used by the mini game."""
    outcome_a = score_patient(profile_a)
    if isinstance(outcome_a, Fail):
        return fail(outcome_a.error)
    outcome_b = score_patient(profile_b)
    if isinstance(outcome_b, Fail):
        return fail(outcome_b.error)
    assert isinstance(outcome_a, Ok) and isinstance(outcome_b, Ok)
    return ok((outcome_a.value, outcome_b.value))


def feature_semantics_labels() -> list[str]:
    return list(FEATURE_SEMANTICS)
