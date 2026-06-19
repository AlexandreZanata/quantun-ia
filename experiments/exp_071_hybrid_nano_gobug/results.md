# Results — EXP 071: Hybrid QNN Head on Frozen GoBug LargeNanoMLP (C3)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC)

- Frozen backbone params: **1,131,072**
- Trainable head params: **289**
- Train rows: **27,172**
- Val rows: **5,822**
- Classical head PR-AUC: **0.3174**
- Hybrid head PR-AUC: **0.3175**
- Δ vs classical: **0.02 pp**
- Elapsed: **8.797s**

## Verdict
**accepted** — hybrid val PR-AUC vs frozen classical head (gate ≥ -1.0 pp).

## Limitations
- Temporal val split only (sha-order proxy); test split untouched.
- QNN sim on CPU; classical backbone on CUDA.
- Software defect benchmark — not production static analysis.
