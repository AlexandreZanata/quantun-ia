# Results — EXP 088 Shadow / Pauli features → NarrowDeepNano

**Profile:** `publication`  
**Verdict:** rejected  
**Train / val rows:** 20000 / 13566  
**Shadow dims:** 64  
**Elapsed:** 498.3s (features 491.6s)

| Model | Val ROC-AUC | Notes |
|-------|-------------|-------|
| LogisticRegression | 0.6960 | raw 37-d |
| NarrowDeepNano (raw) | 0.7610 | 577,665 params |
| NarrowDeepNano (shadow) | 0.6050 | 591,489 params |
| HistGB (honesty) | 0.7990 | not primary gate |

- Δ shadow − classical = **-15.60 pp** (need ≥ -0.5)
- Δ shadow − logistic = **-9.11 pp** (need ≥ 2.0)

## Interpretation

Pauli/shadow feature map did not clear H-Q2.3 gates — do not claim quantum feature advantage on maize.

## Limitations

- Analytic `default.qubit` Pauli expectations (infinite-shot shadow limit).
- Single seed; temporal val only.
