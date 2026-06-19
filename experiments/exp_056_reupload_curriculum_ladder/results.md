# Results — EXP 056: Re-upload depth curriculum ladder

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Wins: **1/3** (gate ≥ **2**)
- Per-rung advantage gate: **≥ 0.3 pp**

| Rung | Metric | Curriculum | Fixed | Δ pp | Won |
|------|--------|------------|-------|------|-----|
| pca_mnist_binary | accuracy | 0.8733 | 0.8067 | +6.67 | yes |
| breast_cancer | accuracy | 0.6257 | 0.9532 | -32.75 | no |
| higgs_50k | roc_auc | 0.7201 | 0.7274 | -0.73 | no |

- Elapsed: **56.712s**

## Verdict
**honest negative** — re-upload depth curriculum vs fixed max depth.
