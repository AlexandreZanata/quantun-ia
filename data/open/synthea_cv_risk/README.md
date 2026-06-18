# Synthea Cardiovascular Risk Dataset (Tier B)

**Purpose:** Clinical-aligned, million-row **synthetic** tabular data for the Population Cardiovascular Risk Nano Scorer.

## Source

- **Engine:** [Synthea](https://github.com/synthetichealth/synthea) (MIT License)  
- **Output:** FHIR bundles → tabular extractor (`scripts/build_synthea_cv_risk.py`)

## Target cohort (v1)

| Field | Value |
|-------|-------|
| Unit of analysis | Patient-month (eligible adults) |
| Rows | 1,000,000 |
| Features | 40 (demographics, vitals, labs, conditions, meds, utilization) |
| Label | Cardiovascular event within next 12 months (`0` / `1`) |
| PHI | None (synthetic) |

## Splits

| Split | Rows |
|-------|------|
| Train | 700,000 |
| Val | 150,000 |
| Test | 150,000 |

Stratified on `label`, `random_state=42`.

## Prerequisites

- Java 11+ (`java -version`)
- Optional: Synthea JAR for FHIR extraction path
- ~2 GB free disk for processed parquet (simulation mode)
- ~15 GB if generating raw FHIR via Synthea CLI

## Build modes

### Default — clinical simulation (RTX 4060 gate, ~2 min)

Deterministic epidemiological cohort matching `tabular_binary_v1` (40 features). Documented in `stats.json` as `source_mode: clinical_simulation`.

```bash
source .local/env.sh
make data-open-synthea-cv
make data-open-verify
```

### FHIR extraction — from existing Synthea output

```bash
# After generating FHIR with Synthea CLI (see below)
MLFLOW_DISABLE=1 QML_DEVICE=cuda python scripts/build_synthea_cv_risk.py \
  --fhir-dir data/open/synthea_cv_risk/raw/generated/fhir
```

### Full Synthea CLI path

1. Download `synthea-with-dependencies.jar` from [Synthea releases](https://github.com/synthetichealth/synthea/releases) into `data/open/synthea_cv_risk/raw/`.

2. Generate a patient cohort (example: 120K patients, subsampled to 1M rows):

```bash
java -jar data/open/synthea_cv_risk/raw/synthea-with-dependencies.jar \
  -p 120000 -s 42 Massachusetts
```

3. Build parquet splits from FHIR:

```bash
MLFLOW_DISABLE=1 QML_DEVICE=cuda python scripts/build_synthea_cv_risk.py \
  --fhir-dir data/open/synthea_cv_risk/raw/generated/fhir
```

Or run Synthea inline:

```bash
MLFLOW_DISABLE=1 QML_DEVICE=cuda python scripts/build_synthea_cv_risk.py \
  --run-synthea --population 120000
```

4. Track with DVC:

```bash
dvc add data/open/synthea_cv_risk/processed/v1
```

## Feature semantics

See `stats.json` → `feature_semantics` after build (40 clinical indices mapped to `feature_0 … feature_39`).

## Disclaimer

Synthetic data only — **research prototype**, not for clinical decisions.
