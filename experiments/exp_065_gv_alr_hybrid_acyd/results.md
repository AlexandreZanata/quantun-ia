# Results — EXP 065: GV-ALR on frozen hybrid QNN head (ACYD C4 / H-Q4)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (ROC-AUC)

- Fixed LR: **8** epochs · val ROC-AUC **0.6771** · **27.723s**
- GV-ALR: **5** epochs · val ROC-AUC **0.6763** · **30.009s**
- Δ ROC-AUC: **-0.08 pp**
- Epoch fraction: **5/8**
- Wall-time ratio: **1.08**

## Verdict
**accepted** — |Δ ROC-AUC| ≤ 0.3 pp and adaptive epochs ≤ 70% of fixed.

## Limitations
- PennyLane QNN sim on CPU; frozen C4 backbone from exp_060.
- Val ROC-AUC only; efficiency gate mirrors exp_054/exp_075.
