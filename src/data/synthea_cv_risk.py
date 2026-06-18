"""Synthea cardiovascular risk cohort — FHIR extraction and clinical simulation."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.open_higgs import sha256_file, update_manifest_ready

SYNTHEA_LICENSE = "MIT"
SYNTHEA_SOURCE_URL = "https://github.com/synthetichealth/synthea"
N_FEATURES = 40
COHORT_TOTAL = 1_000_000
TRAIN_ROWS = 700_000
VAL_ROWS = 150_000
TEST_ROWS = 150_000
RANDOM_STATE = 42
LABEL_COLUMN = "label"
FEATURE_COLUMNS = [f"feature_{i}" for i in range(N_FEATURES)]

CV_CONDITION_SNOMED = frozenset({
    "22298006",   # Myocardial infarction
    "230690007",  # Stroke
    "84114007",   # Heart failure
    "49436004",   # Atrial fibrillation
    "53741008",   # Coronary arteriosclerosis
})

FEATURE_SEMANTICS = [
    "age_years_norm",
    "sex_male",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "total_cholesterol",
    "hdl",
    "ldl",
    "triglycerides",
    "glucose",
    "hba1c",
    "creatinine",
    "smoker",
    "diabetes",
    "hypertension",
    "atrial_fibrillation",
    "prior_mi",
    "prior_stroke",
    "copd",
    "ckd_stage",
    "statin",
    "ace_inhibitor",
    "beta_blocker",
    "aspirin",
    "ed_visits_12m",
    "inpatient_12m",
    "outpatient_12m",
    "medication_count",
    "family_history_cvd",
    "framingham_score",
    "ascvd_risk",
    "charlson_index",
    "utilization_index",
    "lab_risk_index",
    "med_burden_index",
    "comorbidity_count",
    "vitals_risk_index",
    "lifestyle_risk_index",
    "latent_risk_factor",
]


def feature_column_names(n_features: int = N_FEATURES) -> list[str]:
    """Return canonical feature column names for tabular_binary_v1."""
    return [f"feature_{i}" for i in range(n_features)]


def simulate_clinical_cohort(
    n_rows: int = COHORT_TOTAL,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic synthetic patient-month rows with CV event labels."""
    rng = np.random.default_rng(random_state)
    age = rng.uniform(30, 90, n_rows).astype(np.float32)
    sex = rng.integers(0, 2, n_rows).astype(np.float32)
    bmi = np.clip(rng.normal(28.0, 5.5, n_rows), 16.0, 55.0).astype(np.float32)
    sbp = np.clip(rng.normal(128.0, 18.0, n_rows), 90.0, 220.0).astype(np.float32)
    dbp = np.clip(rng.normal(78.0, 10.0, n_rows), 50.0, 130.0).astype(np.float32)
    hr = np.clip(rng.normal(74.0, 12.0, n_rows), 45.0, 140.0).astype(np.float32)
    chol = np.clip(rng.normal(200.0, 40.0, n_rows), 120.0, 350.0).astype(np.float32)
    hdl = np.clip(rng.normal(50.0, 12.0, n_rows), 20.0, 100.0).astype(np.float32)
    ldl = np.clip(chol - hdl * 0.4, 50.0, 250.0).astype(np.float32)
    tg = np.clip(rng.normal(150.0, 60.0, n_rows), 50.0, 500.0).astype(np.float32)
    glucose = np.clip(rng.normal(105.0, 25.0, n_rows), 70.0, 300.0).astype(np.float32)
    hba1c = np.clip(rng.normal(5.8, 0.9, n_rows), 4.5, 12.0).astype(np.float32)
    creatinine = np.clip(rng.normal(1.0, 0.3, n_rows), 0.5, 3.5).astype(np.float32)
    smoker = (rng.random(n_rows) < 0.18).astype(np.float32)
    diabetes = (rng.random(n_rows) < 0.12).astype(np.float32)
    hypertension = (sbp > 140).astype(np.float32)
    afib = (rng.random(n_rows) < 0.04).astype(np.float32)
    prior_mi = (rng.random(n_rows) < 0.03).astype(np.float32)
    prior_stroke = (rng.random(n_rows) < 0.025).astype(np.float32)
    copd = (rng.random(n_rows) < 0.08).astype(np.float32)
    ckd = np.clip(rng.integers(0, 4, n_rows), 0, 3).astype(np.float32)
    statin = (ldl > 130).astype(np.float32) * (rng.random(n_rows) < 0.7).astype(np.float32)
    ace = hypertension * (rng.random(n_rows) < 0.55).astype(np.float32)
    beta = prior_mi * (rng.random(n_rows) < 0.6).astype(np.float32)
    aspirin = (prior_mi + prior_stroke > 0).astype(np.float32) * (
        rng.random(n_rows) < 0.75
    ).astype(np.float32)
    ed_visits = np.clip(rng.poisson(0.4, n_rows), 0, 10).astype(np.float32)
    inpatient = np.clip(rng.poisson(0.15, n_rows), 0, 5).astype(np.float32)
    outpatient = np.clip(rng.poisson(4.0, n_rows), 0, 30).astype(np.float32)
    med_count = np.clip(rng.poisson(3.5, n_rows), 0, 20).astype(np.float32)
    family_hx = (rng.random(n_rows) < 0.22).astype(np.float32)

    framingham = (
        0.03 * age + 0.8 * sex + 0.04 * sbp + 0.02 * chol - 0.03 * hdl + 0.5 * smoker
    ).astype(np.float32)
    ascvd = (0.025 * age + 0.6 * diabetes + 0.4 * hypertension + 0.7 * prior_mi).astype(
        np.float32
    )
    charlson = (
        diabetes + copd + ckd + prior_mi * 2 + prior_stroke * 1.5 + afib
    ).astype(np.float32)
    utilization = (ed_visits + inpatient * 3 + outpatient * 0.1).astype(np.float32)
    lab_risk = ((glucose - 100) / 40 + (hba1c - 5.7) / 1.5 + (creatinine - 1.0) / 0.5).astype(
        np.float32
    )
    med_burden = (statin + ace + beta + aspirin + med_count * 0.1).astype(np.float32)
    comorbidity = (diabetes + hypertension + copd + afib + prior_mi + prior_stroke).astype(
        np.float32
    )
    vitals_risk = ((sbp - 120) / 20 + (bmi - 25) / 10 + (hr - 70) / 20).astype(np.float32)
    lifestyle = (smoker + (bmi > 30).astype(np.float32) + family_hx * 0.5).astype(np.float32)
    latent = rng.normal(0.0, 1.0, n_rows).astype(np.float32)

    logit = (
        -4.2
        + 0.035 * age
        + 0.7 * sex
        + 0.04 * sbp
        + 0.015 * ldl
        + 0.9 * diabetes
        + 0.6 * smoker
        + 1.2 * prior_mi
        + 1.0 * prior_stroke
        + 0.5 * afib
        + 0.08 * latent
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.random(n_rows) < prob).astype(np.float32)

    features = np.column_stack(
        [
            age / 100.0,
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
    ).astype(np.float32)
    return features, labels


def _patient_age_years(patient: dict[str, Any], reference: datetime) -> float:
    birth = patient.get("birthDate")
    if not birth:
        return 55.0
    born = datetime.strptime(birth[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    return max(18.0, (reference - born).days / 365.25)


def _sex_male(patient: dict[str, Any]) -> float:
    gender = str(patient.get("gender", "")).lower()
    return 1.0 if gender == "male" else 0.0


def _has_cv_condition(conditions: list[dict[str, Any]]) -> bool:
    for item in conditions:
        coding = item.get("code", {}).get("coding", [])
        for code in coding:
            if str(code.get("code", "")) in CV_CONDITION_SNOMED:
                return True
    return False


def _observation_value(observations: list[dict[str, Any]], loinc: str, default: float) -> float:
    for item in observations:
        coding = item.get("code", {}).get("coding", [])
        if not any(str(code.get("code", "")) == loinc for code in coding):
            continue
        value = item.get("valueQuantity", {}).get("value")
        if value is not None:
            return float(value)
    return default


def extract_patient_row(bundle: dict[str, Any], reference: datetime | None = None) -> tuple[np.ndarray, float]:
    """Extract one tabular row from a Synthea FHIR bundle."""
    ref = reference or datetime.now(tz=UTC)
    patient: dict[str, Any] = {}
    conditions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType")
        if rtype == "Patient":
            patient = resource
        elif rtype == "Condition":
            conditions.append(resource)
        elif rtype == "Observation":
            observations.append(resource)

    age = _patient_age_years(patient, ref)
    sex = _sex_male(patient)
    bmi = _observation_value(observations, "39156-5", 28.0)
    sbp = _observation_value(observations, "8480-6", 128.0)
    dbp = _observation_value(observations, "8462-4", 78.0)
    hr = _observation_value(observations, "8867-4", 74.0)
    chol = _observation_value(observations, "2093-3", 200.0)
    hdl = _observation_value(observations, "2085-9", 50.0)
    ldl = max(50.0, chol - hdl * 0.4)
    tg = _observation_value(observations, "2571-8", 150.0)
    glucose = _observation_value(observations, "2339-0", 105.0)
    hba1c = _observation_value(observations, "4548-4", 5.8)
    creatinine = _observation_value(observations, "2160-0", 1.0)
    cv_event = 1.0 if _has_cv_condition(conditions) else 0.0
    diabetes = 1.0 if any(
        "44054006" in str(c.get("code", {})) for c in conditions
    ) else 0.0
    hypertension = 1.0 if sbp >= 140 else 0.0

    row = np.array(
        [
            age / 100.0,
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
            0.0,
            diabetes,
            hypertension,
            0.0,
            cv_event,
            0.0,
            0.0,
            0.0,
            float(ldl > 130),
            float(hypertension),
            0.0,
            float(cv_event),
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            sbp / 20.0,
            age / 50.0,
            diabetes + hypertension,
            1.0,
            glucose / 100.0,
            1.0,
            diabetes + hypertension + cv_event,
            bmi / 30.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    label = cv_event
    return row, label


def load_fhir_cohort(fhir_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load patient rows from Synthea FHIR JSON bundles."""
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for path in sorted(fhir_dir.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        row, label = extract_patient_row(bundle)
        rows.append(row)
        labels.append(label)
    if not rows:
        msg = f"no FHIR bundles found in {fhir_dir}"
        raise FileNotFoundError(msg)
    return np.vstack(rows), np.array(labels, dtype=np.float32)


def subsample_to_target(
    features: np.ndarray,
    labels: np.ndarray,
    n_samples: int = COHORT_TOTAL,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample or oversample with replacement to exactly n_samples rows."""
    if len(labels) == n_samples:
        return features, labels
    if len(labels) > n_samples:
        indices = np.arange(len(labels))
        selected, _ = train_test_split(
            indices,
            train_size=n_samples,
            stratify=labels,
            random_state=random_state,
        )
        return features[selected], labels[selected]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(labels), size=n_samples, replace=True)
    return features[idx], labels[idx]


def split_cohort_partitions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_rows: int = VAL_ROWS,
    test_rows: int = TEST_ROWS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split cohort into stratified train / val / test partitions."""
    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_rows,
        stratify=labels,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rows,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )
    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
        features[test_idx],
        labels[test_idx],
    )


def build_cohort_frame(features: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Build export frame matching tabular_binary_v1 schema."""
    frame = pd.DataFrame(features.astype(np.float32), columns=FEATURE_COLUMNS)
    frame[LABEL_COLUMN] = labels.astype(np.int32)
    return frame


def compute_split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute class balance and feature means for a split."""
    label_counts = frame[LABEL_COLUMN].value_counts().sort_index()
    pos = int(label_counts.get(1, 0))
    neg = int(label_counts.get(0, 0))
    total = len(frame)
    return {
        "n_rows": total,
        "class_counts": {"0": neg, "1": pos},
        "positive_rate": round(pos / total, 6) if total else 0.0,
        "feature_means": {col: float(frame[col].mean()) for col in FEATURE_COLUMNS},
    }


def build_stats_payload(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source_mode: str,
) -> dict[str, Any]:
    """Aggregate stats.json content for processed Synthea CV v1."""
    return {
        "dataset_id": "synthea_cv_risk_v1",
        "license": SYNTHEA_LICENSE,
        "source_url": SYNTHEA_SOURCE_URL,
        "source_mode": source_mode,
        "feature_semantics": FEATURE_SEMANTICS,
        "n_features": N_FEATURES,
        "random_state": RANDOM_STATE,
        "cohort_total": COHORT_TOTAL,
        "splits": {
            "train": compute_split_stats(train),
            "val": compute_split_stats(val),
            "test": compute_split_stats(test),
        },
    }


def write_parquet_splits(
    out_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    source_mode: str,
) -> dict[str, Path]:
    """Write train/val/test parquet files and stats.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    val.to_parquet(paths["val"], index=False)
    test.to_parquet(paths["test"], index=False)
    stats_path = out_dir / "stats.json"
    stats_path.write_text(
        json.dumps(build_stats_payload(train, val, test, source_mode=source_mode), indent=2) + "\n",
        encoding="utf-8",
    )
    paths["stats"] = stats_path
    return paths


def build_synthea_processed(
    out_dir: Path,
    *,
    fhir_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[dict[str, Path], str]:
    """End-to-end build from FHIR or simulated cohort to parquet splits."""
    if fhir_dir is not None and fhir_dir.is_dir():
        features, labels = load_fhir_cohort(fhir_dir)
        features, labels = subsample_to_target(
            features,
            labels,
            n_samples=COHORT_TOTAL,
            random_state=random_state,
        )
        source_mode = "fhir_extraction"
    else:
        features, labels = simulate_clinical_cohort(
            n_rows=COHORT_TOTAL,
            random_state=random_state,
        )
        source_mode = "clinical_simulation"

    x_train, y_train, x_val, y_val, x_test, y_test = split_cohort_partitions(
        features,
        labels,
        random_state=random_state,
    )
    train = build_cohort_frame(x_train, y_train)
    val = build_cohort_frame(x_val, y_val)
    test = build_cohort_frame(x_test, y_test)
    paths = write_parquet_splits(out_dir, train, val, test, source_mode=source_mode)
    return paths, source_mode


def run_synthea_cli(
    jar_path: Path,
    output_dir: Path,
    *,
    population: int,
    seed: int = RANDOM_STATE,
    state: str = "Massachusetts",
) -> Path:
    """Run Synthea JAR to generate FHIR bundles."""
    if not jar_path.is_file():
        msg = f"Synthea JAR not found: {jar_path}"
        raise FileNotFoundError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "java",
            "-jar",
            str(jar_path),
            "-p",
            str(population),
            "-s",
            str(seed),
            state,
        ],
        cwd=output_dir,
        check=True,
    )
    fhir_dir = output_dir / "fhir"
    if not fhir_dir.is_dir():
        msg = f"Synthea did not produce FHIR output at {fhir_dir}"
        raise FileNotFoundError(msg)
    return fhir_dir


def update_synthea_manifest_ready(manifest_path: Path, processed_dir: Path) -> dict[str, Any]:
    """Mark synthea_cv_risk_v1 ready with checksums."""
    return update_manifest_ready(
        manifest_path,
        processed_dir,
        dataset_id="synthea_cv_risk_v1",
    )
