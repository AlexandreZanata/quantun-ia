# Results — EXP 051: Hybrid QNN head on frozen NIHR backbone

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC)

- Frozen backbone params: **1,110,592**
- Trainable head params: **289**
- Train rows: **50,000**
- Val rows: **15,000**
- Classical head PR-AUC: **0.2392**
- Hybrid head PR-AUC: **0.2394**
- Δ vs classical: **0.02 pp**
- Elapsed: **14.582s**

## Verdict
**accepted** — hybrid val PR-AUC vs frozen classical head (gate ≥ -1.0 pp).

## Limitations
- Clinical Spearman gate deferred (exp_041 cases use Synthea feature space).
- QNN sim on CPU; publication uses NIHR val only.
