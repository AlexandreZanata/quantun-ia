# Results — EXP 077: Conventional tabular baselines vs LargeNanoMLP (GoBug code defects)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC primary)

- Train rows: **27,172** | Val rows: **5,822**
- Best conventional val PR-AUC: **0.3276**
- LargeNanoMLP val PR-AUC: **0.3174**
- Advantage: **-1.02 pp** (gate ≥ **0.5** pp)
- Elapsed: **21.623s**

| Model | Val PR-AUC | Val accuracy | Train (s) |
|-------|------------|--------------|-----------|
| HistGradientBoosting (sklearn) | 0.3276 | 0.7224 | 0.2 |
| XGBoost shallow (xgboost) | 0.3192 | 0.7219 | 0.7 |
| LargeNanoMLP (quantun-ia) | 0.3174 | 0.7260 | 0.1 |
| LogisticRegression (sklearn) | 0.3097 | 0.7250 | 0.8 |
| MLPClassifier (sklearn, 2048-512-64) | 0.3039 | 0.7162 | 19.8 |

## Verdict
**rejected** — LargeNanoMLP vs conventional sklearn/XGBoost on GoBug (`profile=publication`).

## Limitations
- Single seed; temporal val split only (aligned with exp_070).
- LargeNanoMLP evaluated from exp_070 checkpoint; baselines retrained each run.
- Software defect benchmark — not production static analysis.
