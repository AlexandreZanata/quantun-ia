# Results — EXP 017 (Poison × Topology)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** circles, label poisoning 0–30%, hybrid topologies + NAS preset

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **nas_preset_poison_0** | 65.2% | [62.9%, 67.3%] |
| quantum_first_poison_0 | 65.1% | [62.4%, 67.8%] |
| classical_first_poison_0 | 61.3% | [59.0%, 63.3%] |
| quantum_first_poison_30 | 60.5% | [57.6%, 63.4%] |
| nas_preset_poison_30 | 60.5% | [57.9%, 63.1%] |
| hybrid_sandwich_poison_0 | 60.5% | [57.9%, 62.7%] |
| classical_first_poison_30 | 59.8% | [56.9%, 62.6%] |
| hybrid_sandwich_poison_30 | 56.7% | [54.3%, 59.1%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_first_poison_0 vs hybrid_sandwich_poison_0 | +4.7 pp | 0.008 | 1.34 | yes |
| quantum_first_poison_30 vs hybrid_sandwich_poison_30 | +3.9 pp | 0.422 | 0.56 | no |
| nas_preset_poison_30 vs hybrid_sandwich_poison_30 | +3.9 pp | 0.422 | 0.61 | no |
| classical_first_poison_30 vs hybrid_sandwich_poison_30 | +3.1 pp | 0.422 | 0.53 | no |

## Conclusion
Topology-driven poisoning robustness on clean holdout. Builds on exp_004/010/016.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
