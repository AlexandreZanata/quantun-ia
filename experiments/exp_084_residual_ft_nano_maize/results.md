# Results — EXP 084: Residual / FT-lite nano vs HistGB (ACYD maize)

**Run date:** 2026-07-12  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Profile:** `publication`

## Validation gate

- Train rows: **151,956** | Val rows: **13,566**
- HistGB val ROC-AUC: **0.8178**
- Best nano: **residual_nano_mlp** AUC **0.8086**
- Advantage: **-0.92 pp** (win ≥ 0.5 · tie ±0.5)
- Elapsed: **57.026s**

| Model | Val ROC-AUC | Params | Train (s) |
|-------|-------------|--------|-----------|
| HistGradientBoosting (sklearn) | 0.8178 | — | — |
| ResidualNanoMLP | 0.8086 | 840,321 | 11.1 |
| NarrowDeepNano | 0.8058 | 577,665 | 9.1 |
| FTLiteNano | 0.5602 | 17,281 | 36.0 |

## Verdict
**rejected** — Phase A H-N1 vs HistGB on ACYD maize.

## Limitations
- Single seed; temporal val only (aligned with exp_081/083).
- Agro-climate benchmark — not operational planting advice.
- If rejected/tie: prefer Phase D distillation (exp_092) before quantum B.
