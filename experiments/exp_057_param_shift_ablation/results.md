# Results — EXP 057: Parameter-shift vs autograd ablation

**Run date:** 2026-06-19  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Validation gate

- Mean holdout gap: **20.99 pp** (gate ≤ **1.0**)
- Variance ratio (autograd/param-shift): **0.08** (gate ≥ **2.0**)

| Method | Mean holdout acc | Grad variance |
|--------|------------------|---------------|
| autograd | 0.8807 | 0.000973 |
| parameter-shift | 0.6708 | 0.012134 |

- Seeds: **10**
- Elapsed: **10348.121s**

## Verdict
**honest negative** — parameter-shift vs autograd on 4q×3L re-upload QNN.
