# Results — EXP 034: LargeNanoMLP on Synthea CV

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Params: **1,165,953**
- Train rows: **700,000**
- Val rows: **150,000**
- Logistic val AUC: **0.7949**
- LargeNanoMLP val AUC: **0.7867**
- Advantage: **-0.82 pp**
- Elapsed: **105.302s**

## Verdict
**accepted** — infrastructure gate (≥ -2.0 pp vs logistic); logistic may win on synthetic EHR (honest negative documented).

## Limitations
- Synthea synthetic EHR — research prototype, not clinical deployment.
- Test split not used for model selection in this gate.
