# Results — EXP 036: HIGGS Methodology Ablation

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Mean val ROC-AUC by method

| Method | Mean val AUC | Δ vs baseline |
|--------|--------------|---------------|
| baseline | **0.7931** | — |
| curriculum | **0.7770** | -1.61 pp |
| adaptive | **0.7957** | +0.26 pp |
| champion | **0.7917** | -0.14 pp |

- Seeds: **10**
- Beaters (≥ 0.5 pp): **none**
- Elapsed: **604.152s**

## Verdict
**rejected (honest negative)** — paired comparison vs Adam baseline on HIGGS slice.
