# Results — EXP 001 (Quantum vs Classical)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| classical_32 | **65.2%** | [63.5%, 67.1%] |
| classical_8 | 59.3% | [57.1%, 61.3%] |
| quantum_reupload_4q_3l | 57.9% | [55.0%, 61.3%] |
| quantum_6q_3l | 54.5% | [51.5%, 57.5%] |
| quantum_4q_2l | 50.3% | [47.8%, 52.1%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| classical_32 vs quantum_4q_2l | +14.9 pp | 0.002 | **yes** |
| reupload vs basic 4q | +7.6 pp | 0.002 | **yes** |
| classical_32 vs reupload | +7.3 pp | 0.002 | **yes** |

## Conclusion
Classical_32 wins decisively. Re-upload closes ~half the gap vs basic QNN but remains significantly below classical.

---

## Publication large (n=1000, same 10 seeds)

| Model | Mean | 95% CI |
|-------|------|--------|
| classical_32 | **68.1%** | [66.1%, 70.2%] |
| classical_8 | 63.7% | [61.0%, 66.4%] |
| quantum_reupload_4q_3l | 62.7% | [59.8%, 65.2%] |
| quantum_6q_3l | 59.5% | [55.3%, 63.8%] |
| quantum_4q_2l | 51.7% | [47.7%, 55.4%] |

CIs are ~25% narrower vs n=500. Same ranking; all three Holm-corrected comparisons remain significant.
