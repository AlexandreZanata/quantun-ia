# Results — EXP 053: Dynamic entanglement schedule (breast cancer)

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Dataset:** Wisconsin breast cancer (UCI), stratified holdout

## Validation gate

- Seeds: **3**
- Train rows: **398** · Holdout rows: **171**
- Schedule stages: **5** × **10** epochs
- Mean schedule holdout: **95.91%**
- Best fixed (none): **96.69%**
- Advantage: **-0.78 pp**
- Paired wins: **1/3**
- Wilcoxon p: **0.75**
- Elapsed: **8.874s**

## Verdict
**honest negative** — dynamic entanglement vs best fixed topology (gate ≥ 1.0 pp).
