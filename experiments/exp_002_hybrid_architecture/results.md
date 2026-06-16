# Results — EXP 002 (Hybrid Architecture)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Holdout results
| Architecture | Mean | 95% CI |
|--------------|------|--------|
| quantum_first | **63.1%** | [60.9%, 65.4%] |
| hybrid_sandwich | 58.7% | [56.7%, 60.9%] |
| classical_first | 56.8% | [55.1%, 58.8%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| quantum_first vs classical_first | +6.3 pp | 0.002 | **yes** |
| hybrid_sandwich vs classical_first | +1.9 pp | 0.219 | no |

## Conclusion
QuantumFirst beats ClassicalFirst significantly but still trails classical_32 from exp_001 (65.2%). Hybrids do not beat dedicated classical MLP.
