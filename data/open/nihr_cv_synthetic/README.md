# NIHR Synthetic Cardiovascular Dataset

**Purpose:** Realistic-prevalence synthetic primary-care cohort for cardiovascular event classification (Phase 3 / exp_044).

## Source

- **Zenodo:** [10.5281/zenodo.12567416](https://doi.org/10.5281/zenodo.12567416) (CC0-1.0)
- **File:** `cvd_synthetic_dataset_v0.2.csv` (~100k rows)
- **Builder:** `scripts/build_nihr_cv_synthetic.py`

## Cohort (v1)

| Field | Value |
|-------|-------|
| Rows | 100,000 |
| Features | 13 (demographics, vitals, comorbidities, FEV1) |
| Label | `heart_attack_or_stroke_occurred` (10-year event) |
| Prevalence | ~6.6% positive |
| Missingness | BMI, SBP, FEV1 — train-median imputed |

## Splits

| Split | Rows |
|-------|------|
| Train | 70,000 |
| Val | 15,000 |
| Test | 15,000 |

Stratified on `label`, `random_state=42`.

## Build

```bash
source .local/env.sh
make data-open-nihr-cv
make data-open-verify
```

## Citation

Burns, D., Richardson, K., Driessens, C. A synthetic dataset for survival and classification models: prediction of heart attack or stroke. Zenodo (2024). https://doi.org/10.5281/zenodo.12567416
