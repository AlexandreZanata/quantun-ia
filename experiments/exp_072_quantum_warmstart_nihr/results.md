# Results — EXP 072: Quantum warm-start on NIHR hybrid (C2 replication)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate (PR-AUC)

- Seeds: **3**
- Train rows: **50,000** · Val rows: **15,000**
- Schedule: **7** classical + **3** quantum epochs
- Mean e2e PR-AUC: **0.2343**
- Mean warm-start PR-AUC: **0.2307**
- Advantage: **-0.35 pp**
- Paired wins: **1/3**
- Wilcoxon p: **0.5**
- Elapsed: **87.675s**

## Verdict
**honest negative** — warm-start val PR-AUC vs end-to-end hybrid (gate ≥ 0.5 pp).

## Limitations
- HybridSandwich protocol (mirrors exp_052); not frozen C2 backbone.
- QNN sim on CPU; val PR-AUC only.
