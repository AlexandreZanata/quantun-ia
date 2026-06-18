# Results — EXP 032: LargeNanoMLP on HIGGS

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Params: **1,141,377**
- Train rows: **805,000**
- Val rows: **172,500**
- Logistic val AUC: **0.6849**
- LargeNanoMLP val AUC: **0.8258**
- Advantage: **14.09 pp**
- Elapsed: **108.402s**

## Verdict
**accepted** — val AUC beats logistic by ≥ 1.0 pp.

## Limitations
- HIGGS physics tabular — infrastructure validation, not clinical claim.
- Test split not used for model selection in this gate.
