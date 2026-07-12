# Results — EXP 064: Dynamic entanglement schedule on ACYD (C4 / H-Q3)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Dataset:** `acyd_soy_brazil_v1` (val ROC-AUC)

## Validation gate (ROC-AUC)

- Seeds: **3**
- Train rows: **10,000** · Val rows: **3,000**
- Schedule stages: **5** × **10** epochs
- Mean schedule val: **0.6314**
- Best fixed (none): **0.6422**
- Advantage: **-1.08 pp**
- Paired wins: **1/3**
- Wilcoxon p: **0.75**
- Elapsed: **53.115s**

## Verdict
**honest negative** — dynamic entanglement vs best fixed topology (gate ≥ 0.5 pp).

## Limitations
- Standalone `QuantumNetEntangled` (mirrors exp_053/exp_074); not frozen C4 backbone.
- PennyLane QNN sim on CPU; publication row cap for feasible epoch cost.
