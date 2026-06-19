# Results — EXP 062: Hybrid QNN Head on Frozen ACYD LargeNanoMLP (C4)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (ROC-AUC)

- Frozen backbone params: **1,159,744**
- Trainable head params: **289**
- Train rows: **50,107**
- Val rows: **5,830**
- Classical head val ROC-AUC: **0.6777**
- Hybrid head val ROC-AUC: **0.6758**
- Δ vs classical: **-0.19 pp**
- Elapsed: **14.72s**

## Verdict
**accepted** — hybrid val ROC-AUC vs frozen classical head (gate ≥ -1.0 pp).

## Limitations
- Temporal val split only (crop-years 2019–2021); test years untouched.
- QNN sim on CPU; classical backbone on CUDA.
- Agro-climate benchmark — not operational planting advice.
