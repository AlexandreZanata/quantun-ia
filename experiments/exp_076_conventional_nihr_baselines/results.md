# Results — EXP 076: Conventional tabular baselines vs LargeNanoMLP (NIHR synthetic CV)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC primary)

- Train rows: **70,000** | Val rows: **15,000**
- Best conventional val PR-AUC: **0.2382**
- LargeNanoMLP val PR-AUC: **0.2393**
- Advantage: **+0.12 pp** (gate ≥ **0.5** pp)
- Elapsed: **39.395s**

| Model | Val PR-AUC | Val accuracy | Train (s) |
|-------|------------|--------------|-----------|
| LargeNanoMLP (quantun-ia) | 0.2393 | 0.9339 | 0.1 |
| LogisticRegression (sklearn) | 0.2382 | 0.9336 | 0.2 |
| XGBoost shallow (xgboost) | 0.2344 | 0.9338 | 0.7 |
| MLPClassifier (sklearn, 2048-512-64) | 0.2327 | 0.9339 | 38.0 |
| HistGradientBoosting (sklearn) | 0.2304 | 0.9337 | 0.2 |

## Verdict
**rejected** — LargeNanoMLP vs conventional sklearn/XGBoost on NIHR (`profile=publication`).

## Limitations
- Single seed; val split only (aligned with exp_069).
- LargeNanoMLP evaluated from exp_069 checkpoint; baselines retrained each run.
- Clinical synthetic benchmark — not real patient data.
