# Results — EXP 068b: Compound Stress Label on ACYD (H-Q12)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (ROC-AUC vs logistic)

- Train rows: **50,107**
- Val rows: **5,830**
- Train positive rate: **0.0545**
- Val positive rate: **0.1842**
- Logistic val ROC-AUC: **0.8462**
- Hybrid val ROC-AUC: **0.8074**
- Δ vs logistic: **-3.88 pp**
- Elapsed: **47.833s**

## Verdict
**honest negative** — compound-stress hybrid vs logistic (gate ≥ 1.0 pp).

## Limitations
- Drought stress via train-fitted precipitation z-score (SPEI proxy).
- Temporal val only; highly imbalanced compound label.
- QNN sim on CPU; classical backbone on CUDA.
