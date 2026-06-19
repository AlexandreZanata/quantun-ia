# Results — EXP 073: Quantum warm-start on GoBug hybrid (C3 replication)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC)

- Seeds: **3**
- Train rows: **27,172** · Val rows: **5,822**
- Schedule: **7** classical + **3** quantum epochs
- Mean e2e PR-AUC: **0.3032**
- Mean warm-start PR-AUC: **0.3067**
- Advantage: **+0.35 pp**
- Paired wins: **2/3**
- Wilcoxon p: **0.75**
- Elapsed: **45.145s**

## Verdict
**honest negative** — warm-start val PR-AUC vs end-to-end hybrid (gate ≥ 0.5 pp).

## Limitations
- HybridSandwich protocol (mirrors exp_052); not frozen C3 backbone.
- QNN sim on CPU; val PR-AUC only.
