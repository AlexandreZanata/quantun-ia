# Results — EXP 042: Sample-Scale Precision Curve

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Model:** `exp_034` LargeNanoMLP · `synthea_cv_risk_v1` · seed 42

## Summary

| Metric | Value | Gate |
|--------|-------|------|
| Sample sizes | **100 → 2000 (step 100)** | 20 points |
| ROC-AUC @ n=2000 | **0.8100** | ≥ 0.78 |
| 100-row negatives / positives | **10 / 90** | min 10 neg |
| 100-row accuracy | **0.9000** | — |
| 100-row precision | **0.9000** | — |
| 100-row recall | **1.0000** | — |
| 100-row F1 | **0.9474** | — |
| 100-row ROC-AUC | **0.6644** | — |
| Elapsed | **5.9s** | — |

## Verdict
**accepted** — holdout metrics stable across sample sizes.

## Sample-scale curve

| n | Neg | Pos | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|-----|-----|----------|-----------|--------|-----|---------|--------|-------|-----|
| 100 | 10 | 90 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.6644 | 0.9490 | 0.0988 | 0.0969 |
| 200 | 1 | 199 | 0.9950 | 0.9950 | 1.0000 | 0.9975 | 0.4372 | 0.9959 | 0.0050 | 0.0016 |
| 300 | 1 | 299 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.4415 | 0.9973 | 0.0034 | 0.0003 |
| 400 | 1 | 399 | 0.9975 | 0.9975 | 1.0000 | 0.9987 | 0.4637 | 0.9981 | 0.0025 | 0.0010 |
| 500 | 2 | 498 | 0.9960 | 0.9960 | 1.0000 | 0.9980 | 0.6476 | 0.9982 | 0.0040 | 0.0008 |
| 600 | 2 | 598 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.6421 | 0.9984 | 0.0033 | 0.0001 |
| 700 | 2 | 698 | 0.9971 | 0.9971 | 1.0000 | 0.9986 | 0.6433 | 0.9986 | 0.0029 | 0.0003 |
| 800 | 3 | 797 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7440 | 0.9988 | 0.0037 | 0.0007 |
| 900 | 3 | 897 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7503 | 0.9989 | 0.0033 | 0.0003 |
| 1000 | 4 | 996 | 0.9960 | 0.9960 | 1.0000 | 0.9980 | 0.7924 | 0.9989 | 0.0040 | 0.0010 |
| 1100 | 4 | 1096 | 0.9964 | 0.9964 | 1.0000 | 0.9982 | 0.7913 | 0.9990 | 0.0036 | 0.0006 |
| 1200 | 4 | 1196 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7906 | 0.9991 | 0.0033 | 0.0003 |
| 1300 | 5 | 1295 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7946 | 0.9990 | 0.0038 | 0.0009 |
| 1400 | 5 | 1395 | 0.9964 | 0.9964 | 1.0000 | 0.9982 | 0.7950 | 0.9991 | 0.0036 | 0.0006 |
| 1500 | 5 | 1495 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7956 | 0.9991 | 0.0033 | 0.0004 |
| 1600 | 6 | 1594 | 0.9962 | 0.9962 | 1.0000 | 0.9981 | 0.7824 | 0.9990 | 0.0037 | 0.0008 |
| 1700 | 6 | 1694 | 0.9965 | 0.9965 | 1.0000 | 0.9982 | 0.7848 | 0.9991 | 0.0035 | 0.0006 |
| 1800 | 6 | 1794 | 0.9967 | 0.9967 | 1.0000 | 0.9983 | 0.7844 | 0.9991 | 0.0033 | 0.0004 |
| 1900 | 7 | 1893 | 0.9963 | 0.9963 | 1.0000 | 0.9982 | 0.8093 | 0.9991 | 0.0037 | 0.0007 |
| 2000 | 7 | 1993 | 0.9965 | 0.9965 | 1.0000 | 0.9982 | 0.8100 | 0.9992 | 0.0035 | 0.0006 |

## Exported artifacts (local workstation)

- `.local/out/predictions_100_synthea_cv.json` — 100 scored val rows
- `.local/out/sample_scale_precision.json` — full curve JSON

## Interpretation

- Synthea v1 val split has ~99% positive prevalence → accuracy is inflated.
- **Precision / recall / F1** and **ROC-AUC** are the primary metrics for paper tables.
- Each n uses balanced subsampling with **min 10 negatives** (seed 42).
- At n=100: 10 negatives + 90 positives — ROC-AUC and PR-AUC are meaningful.

## Limitations

- Synthetic data; not calibrated for real-world prevalence.
- Subsample metrics have sampling variance at n=100.
