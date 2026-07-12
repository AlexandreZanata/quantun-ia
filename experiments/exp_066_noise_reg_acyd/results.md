# Results — EXP 066: Depolarizing noise on ACYD hybrid QNN (H-Q10 / H-Q5)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Eval split:** temporal test (years ≥ 2022)

## Validation gate (ROC-AUC)

- Depolarizing p: **0.03**
- Train rows: **50,107** · Val: **5,830** · Test: **5,856**
- Noiseless test ROC-AUC: **0.6293**
- Noisy test ROC-AUC: **0.6392**
- Advantage: **+0.99 pp**
- Elapsed: **300.562s**

## Verdict
**accepted** — noisy vs noiseless hybrid on temporal test (gate ≥ 0.5 pp).
