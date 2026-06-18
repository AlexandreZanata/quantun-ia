# Results — EXP 040: Full-Scale HIGGS Methodology Ablation

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Mean val ROC-AUC by method (805K train)

| Method | Mean val AUC | Δ vs baseline |
|--------|--------------|---------------|
| baseline | **0.8355** | — |
| curriculum | **0.8403** | +0.48 pp |
| adaptive | **0.8213** | -1.41 pp |
| champion | **0.8282** | -0.72 pp |

- Train rows: **805,000**
- Val rows: **172,500**
- Seeds: **3**
- Beaters (≥ 0.5 pp): **none**
- Elapsed: **1975.343s**

## Verdict
**rejected (honest negative)** — full-scale paired comparison vs Adam baseline on HIGGS.

## Comparison to exp_036 (50K slice)
- exp_036 best alternative: adaptive **+0.26 pp** (10 seeds, honest negative).
