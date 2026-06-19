# Results — EXP 074: Dynamic entanglement schedule on NIHR (C2 replication)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Dataset:** `nihr_cv_synthetic_v1` (val PR-AUC)

## Validation gate (PR-AUC)

- Seeds: **3**
- Train rows: **10,000** · Val rows: **3,000**
- Schedule stages: **5** × **10** epochs
- Mean schedule val: **0.1963**
- Best fixed (ring): **0.2329**
- Advantage: **-3.66 pp**
- Paired wins: **0/3**
- Wilcoxon p: **0.25**
- Elapsed: **50.904s**

## Verdict
**honest negative** — dynamic entanglement vs best fixed topology (gate ≥ 0.5 pp).

## Limitations
- Standalone `QuantumNetEntangled` (mirrors exp_053); not frozen C2 backbone.
- PennyLane QNN sim on CPU; publication row cap for feasible epoch cost.
