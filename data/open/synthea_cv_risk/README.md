# Synthea Cardiovascular Risk Dataset (Tier B)

**Purpose:** Clinical-aligned, million-row **synthetic** tabular data for the Population Cardiovascular Risk Nano Scorer.

## Source

- **Engine:** [Synthea](https://github.com/synthetichealth/synthea) (MIT License)  
- **Output:** FHIR bundles → custom tabular extractor (`scripts/build_synthea_cv_risk.py`, Phase L3)

## Target cohort (v1)

| Field | Value |
|-------|-------|
| Unit of analysis | Patient-month (eligible adults) |
| Rows | 1,000,000 |
| Features | ~40 (age, sex, BMI, conditions, meds, labs, utilization) |
| Label | Cardiovascular event within next 12 months (`0` / `1`) |
| PHI | None (synthetic) |

## Splits

| Split | Rows |
|-------|------|
| Train | 700,000 |
| Val | 150,000 |
| Test | 150,000 |

## Prerequisites

- Java 11+  
- Synthea JAR or Gradle build  
- ~15 GB free disk for raw FHIR + parquet  

## Build (when script lands)

```bash
make data-open-synthea-cv
```

## Disclaimer

Synthetic data only — **research prototype**, not for clinical decisions.
