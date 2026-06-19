# Results — EXP 069: LargeNanoMLP on NIHR synthetic CV (C2 anchor)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC primary)

- Params: **1,110,657**
- Train rows: **70,000**
- Val rows: **15,000**
- Logistic val PR-AUC: **0.2382**
- LargeNanoMLP val PR-AUC: **0.2370**
- Advantage: **-0.12 pp**
- Elapsed: **6.224s**

## Verdict
**rejected** — val PR-AUC beats logistic by ≥ 1.0 pp.

## Limitations
- Synthetic NIHR cohort (CC0 Zenodo) — not clinical deployment.
- Train-only median imputation; calibrated head deferred.
