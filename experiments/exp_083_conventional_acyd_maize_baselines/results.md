# Results — EXP 083: Conventional tabular baselines vs LargeNanoMLP (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- Best conventional val ROC-AUC: **0.8178**
- LargeNanoMLP val ROC-AUC: **0.8086**
- Advantage: **-0.92 pp** (gate ≥ **0.5** pp)
- Elapsed: **165.618s**

| Model | Val ROC-AUC | Val accuracy | Train (s) |
|-------|-------------|--------------|-----------|
| HistGradientBoosting (sklearn) | 0.8178 | 0.7439 | 1.3 |
| LargeNanoMLP (quantun-ia) | 0.8086 | 0.7291 | 0.1 |
| MLPClassifier (sklearn, 2048-512-64) | 0.8018 | 0.7234 | 158.4 |
| XGBoost shallow (xgboost) | 0.7706 | 0.7055 | 1.2 |
| LogisticRegression (sklearn) | 0.6983 | 0.6421 | 4.5 |

## Verdict
**rejected** — LargeNanoMLP vs conventional sklearn/XGBoost on ACYD maize (`profile=publication`).

## Limitations
- Single seed; temporal val split only (aligned with exp_081).
- LargeNanoMLP evaluated from exp_081 checkpoint; baselines retrained each run.
- Agro-climate benchmark — not operational planting advice.
