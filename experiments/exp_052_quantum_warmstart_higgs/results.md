# Results — EXP 052: Quantum warm-start on HIGGS hybrid

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Seeds: **3**
- Train rows: **50,000** · Val rows: **10,000**
- Schedule: **7** classical + **3** quantum epochs
- Mean e2e AUC: **0.7541**
- Mean warm-start AUC: **0.7499**
- Advantage: **-0.42 pp**
- Paired wins: **1/3**
- Wilcoxon p: **0.5**
- Elapsed: **85.615s**

## Verdict
**honest negative** — warm-start val ROC-AUC vs end-to-end hybrid (gate ≥ 0.5 pp).
