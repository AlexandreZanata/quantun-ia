# Results — EXP 068a: Seasonal Angle Encoding on ACYD (H-Q8)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gates (ROC-AUC)

- Classical head val ROC-AUC: **0.6777**
- Angle seasonal val ROC-AUC: **0.4979**
- Amplitude seasonal val ROC-AUC: **0.5137**
- Angle vs classical: **-17.98 pp**
- Angle vs amplitude: **-1.58 pp**
- Elapsed: **19.029s**

## Verdict
**honest negative** — seasonal angle encoding vs classical and amplitude baselines.

## Limitations
- Cyclic features from scaled in-season weather means (37-dim parquet).
- Single seed publication; QNN sim on CPU.
