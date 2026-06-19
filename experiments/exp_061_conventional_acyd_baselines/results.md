# Results — EXP 061: Conventional tabular baselines vs LargeNanoMLP (ACYD soybean)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Train rows: **50,107** | Val rows: **5,830**
- Best conventional val ROC-AUC: **0.6941**
- LargeNanoMLP val ROC-AUC: **0.6777**
- Advantage: **-1.64 pp** (gate ≥ **0.5** pp)
- Elapsed: **33.779s**

| Model | Val ROC-AUC | Val accuracy | Train (s) |
|-------|-------------|--------------|-----------|
| HistGradientBoosting (sklearn) | 0.6941 | 0.6468 | 0.4 |
| XGBoost shallow (xgboost) | 0.6882 | 0.6197 | 0.7 |
| LargeNanoMLP (quantun-ia) | 0.6777 | 0.6266 | 0.1 |
| MLPClassifier (sklearn, 2048-512-64) | 0.6736 | 0.6346 | 29.6 |
| LogisticRegression (sklearn) | 0.6391 | 0.5983 | 2.9 |

## Verdict
**rejected** — LargeNanoMLP vs conventional sklearn/XGBoost on ACYD (`profile=publication`).

## Limitations
- Single seed; temporal val split only (aligned with exp_060).
- LargeNanoMLP evaluated from exp_060 checkpoint; baselines retrained each run.
- Agro-climate benchmark — not operational planting advice.
