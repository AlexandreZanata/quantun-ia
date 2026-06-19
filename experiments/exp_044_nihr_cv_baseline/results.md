# Results — EXP 044: NIHR synthetic CV baseline

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Holdout metrics

| Model | Val ROC-AUC |
|-------|-------------|
| Logistic (QRISK-style) | 0.8322 |
| LargeNanoMLP | 0.8306 |

- Train rows: **70,000**
- Val rows: **15,000**
- Params: **1,110,657**
- nano − logistic: **-0.16 pp**
- Wall time: **7.531s**

## Verdict
**rejected / inconclusive**

## Limitations
- Synthetic NIHR cohort (CC0 Zenodo) — not clinical deployment.
- Train-only median imputation; calibrated head deferred to exp_047.
