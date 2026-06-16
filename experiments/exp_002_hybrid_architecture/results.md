# Results — EXP 002 (Hybrid Architecture)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, **re-upload QNN (4q, 3 layers)**

## Holdout results
| Architecture | Mean | 95% CI |
|--------------|------|--------|
| quantum_first | **65.7%** | [62.7%, 68.7%] |
| hybrid_sandwich | 62.5% | [59.9%, 64.8%] |
| classical_first | 61.1% | [57.9%, 64.4%] |

## Paired Wilcoxon (Holm-Bonferroni)
| Comparison | Mean diff | p-value | p_holm | Significant |
|------------|-----------|---------|--------|-------------|
| quantum_first vs classical_first | +4.6 pp | 0.006 | **0.012** | **yes** |
| hybrid_sandwich vs classical_first | +1.3 pp | 0.385 | 0.385 | no |

## Conclusion
Re-upload baseline lifts QuantumFirst to **65.7%**, matching classical_32 from exp_001 (65.2%). QuantumFirst significantly beats ClassicalFirst after Holm correction. Hybrids still do not beat the dedicated classical MLP from exp_001.
