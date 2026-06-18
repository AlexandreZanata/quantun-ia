# HIGGS Bootstrap Dataset (Tier A)

**Purpose:** Fast, legal, million-row tabular data to validate the Phase L training pipeline on RTX 4060 before Synthea export.

## Source

- **Name:** HIGGS Data Set  
- **URL:** https://archive.ics.uci.edu/ml/datasets/HIGGS  
- **License:** CC0 1.0 (public domain dedication)  
- **Rows (full):** 11,000,000  
- **Features:** 28 kinematic properties  
- **Label:** 1 = Higgs signal, 0 = background  

## Local subset (v1)

| Split | Rows | Fraction |
|-------|------|----------|
| Train | 805,000 | 70% |
| Val | 172,500 | 15% |
| Test | 172,500 | 15% |
| **Total** | **1,150,000** | 100% |

Stratified on `label`, `random_state=42`.

## Output layout (after build)

```
processed/v1/
├── train.parquet
├── val.parquet
├── test.parquet
└── stats.json
```

## Build

```bash
make data-open-higgs
```

## Why HIGGS first?

- Not clinical — use only for **infrastructure** (OOM checks, batch parity, million-param training).
- Clinical narrative resumes with `synthea_cv_risk_v1` (Tier B).
