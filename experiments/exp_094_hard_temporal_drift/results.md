# Results — EXP 094 Hard temporal drift (ACYD maize)

**Profile:** `publication`  
**Verdict:** accepted  
**Split:** train ≤ 2016 · val [2017, 2018] · test ≥ 2022  
**Rows:** train=142,926 val=9,030 test=13,537  
**Elapsed:** 35.6s

| Model | Val ROC-AUC | Notes |
|-------|-------------|-------|
| HistGB | 0.8246 | hard-drift val |
| ResidualNanoMLP | 0.8185 | 840,321 params |

- Δ nano − HistGB = **-0.61 pp** (need ≥ -1.0)

## Interpretation

ResidualNano stayed within the hard-drift gate vs HistGB.

## Limitations

- Rebuild from raw ACYD maize (standard parquet has no year).
- Single seed; agro research benchmark.
