# Results — EXP 054: GV-ALR on frozen hybrid QNN head

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Fixed LR: **8** epochs · val AUC **0.8327** · **15.271s**
- GV-ALR: **5** epochs · val AUC **0.8328** · **8.945s**
- Δ AUC: **0.01 pp**
- Epoch fraction: **5/8**
- Wall-time ratio: **0.59**

## Verdict
**accepted** — |ΔAUC| ≤ 0.3 pp and adaptive epochs ≤ 70% of fixed.
