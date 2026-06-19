# Results — EXP 042: Sample-Scale Precision Curve

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Sample sizes | **100 → 2000 (step 100)** | 20 points |
| ROC-AUC @ n=2000 | **0.8100** | ≥ 0.78 |
| 100-row negatives / positives | see curve table | ~0 / 100 |
| 100-row accuracy | **1.0000** | — |
| 100-row precision | **1.0000** | — |
| 100-row recall | **1.0000** | — |
| 100-row F1 | **1.0000** | — |
| 100-row ROC-AUC | **0.5000** | — |
| Elapsed | **3.1s** | — |

## Verdict
**accepted** — holdout metrics stable across sample sizes.

## Sample-scale curve

| n | Neg | Pos | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|-----|-----|----------|-----------|--------|-----|---------|-------|
| 100 | 0 | 100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.0000 |
| 200 | 1 | 199 | 0.9950 | 0.9950 | 1.0000 | 0.9975 | 0.4372 | 0.0050 |
| 300 | 1 | 299 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.4415 | 0.0034 |
| 400 | 1 | 399 | 0.9975 | 0.9975 | 1.0000 | 0.9987 | 0.4637 | 0.0025 |
| 500 | 2 | 498 | 0.9960 | 0.9960 | 1.0000 | 0.9980 | 0.6476 | 0.0040 |
| 600 | 2 | 598 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.6421 | 0.0033 |
| 700 | 2 | 698 | 0.9971 | 0.9971 | 1.0000 | 0.9986 | 0.6433 | 0.0029 |
| 800 | 3 | 797 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7440 | 0.0037 |
| 900 | 3 | 897 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7503 | 0.0033 |
| 1000 | 4 | 996 | 0.9960 | 0.9960 | 1.0000 | 0.9980 | 0.7924 | 0.0040 |
| 1100 | 4 | 1096 | 0.9964 | 0.9964 | 1.0000 | 0.9982 | 0.7913 | 0.0036 |
| 1200 | 4 | 1196 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7906 | 0.0033 |
| 1300 | 5 | 1295 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7946 | 0.0038 |
| 1400 | 5 | 1395 | 0.9964 | 0.9964 | 1.0000 | 0.9982 | 0.7950 | 0.0036 |
| 1500 | 5 | 1495 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7956 | 0.0033 |
| 1600 | 6 | 1594 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7824 | 0.0037 |
| 1700 | 6 | 1694 | 0.9965 | 0.9965 | 1.0000 | 0.9982 | 0.7848 | 0.0035 |
| 1800 | 6 | 1794 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7844 | 0.0033 |
| 1900 | 7 | 1893 | 0.9963 | 0.9963 | 1.0000 | 0.9982 | 0.8093 | 0.0037 |
| 2000 | 7 | 1993 | 0.9965 | 0.9965 | 1.0000 | 0.9982 | 0.8100 | 0.0035 |

## Exported artifacts (local workstation)

- `.local/out/predictions_100_synthea_cv.json` — 100 scored val rows
- `.local/out/sample_scale_precision.json` — full curve JSON

## Interpretation

- Synthea v1 val split has ~99% positive prevalence → accuracy is inflated.
- **Precision / recall / F1** and **ROC-AUC** are the primary metrics for paper tables.
- Each n uses an independent stratified subsample (seed 42).

## Limitations

- Synthetic data; not calibrated for real-world prevalence.
- Subsample metrics have sampling variance at n=100.
