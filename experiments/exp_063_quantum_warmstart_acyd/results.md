# Results — EXP 063: Quantum warm-start on ACYD hybrid (C4 / H-Q9)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (ROC-AUC)

- Seeds: **3**
- Train rows: **50,107** · Val rows: **5,830**
- Schedule: **7** classical + **3** quantum epochs
- Mean e2e AUC: **0.6574**
- Mean warm-start AUC: **0.6680**
- Advantage: **+1.06 pp**
- Paired wins: **3/3**
- Wilcoxon p: **0.25**
- Elapsed: **156.98s**

## Verdict
**accepted** — warm-start val ROC-AUC vs end-to-end hybrid (gate ≥ 0.5 pp).

## Limitations
- HybridSandwich protocol (mirrors exp_052); not frozen C4 backbone.
- PennyLane QNN sim on CPU; val ROC-AUC only.
