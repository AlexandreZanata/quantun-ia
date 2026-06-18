# Results — EXP 037: Hybrid QNN Head on Frozen LargeNanoMLP

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Frozen backbone params: **1,141,312**
- Trainable head params: **289**
- Train rows: **50,000**
- Val rows: **10,000**
- Classical head val AUC: **0.8324**
- Hybrid head val AUC: **0.8328**
- Δ vs classical: **0.04 pp**
- Elapsed: **16.684s**

## Verdict
**accepted** — hybrid val AUC vs frozen classical head (gate ≥ -1.0 pp).

## Limitations
- QNN sim on CPU; CI slice smaller than full 805K train.
- Test split not used for model selection.
