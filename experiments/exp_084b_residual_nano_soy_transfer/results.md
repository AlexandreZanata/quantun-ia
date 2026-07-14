# Results — EXP 084b: ResidualNano soy transfer vs HistGB

**Run date:** 2026-07-14  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **50,107** | Val rows: **5,830**
- Params: **840,321**
- HistGB val ROC-AUC: **0.6941**
- ResidualNanoMLP val ROC-AUC: **0.6740**
- Advantage: **-2.01 pp** (win ≥ 0.5 · tie ±0.5)
- Elapsed: **6.814s**

## Verdict
**rejected** — Phase A A-T4 ResidualNano soy transfer.

## Limitations
- Single seed; from-scratch soy train (architecture transfer, not weight transfer).
- Hyperparameters frozen from exp_084 ResidualNano maize recipe.
- Agro-climate benchmark — not operational planting advice.
