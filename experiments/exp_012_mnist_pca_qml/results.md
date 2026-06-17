# Results — EXP 012 (MNIST PCA QML)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** MNIST 0 vs 1, PCA-8, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **quantum_amplitude** | 99.9% | [99.8%, 100.0%] |
| quantum_angle | 93.3% | [90.1%, 96.4%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_amplitude vs quantum_angle | +6.7 pp | 0.004 | 1.22 | yes |

## Conclusion
Angle vs amplitude encoding on PCA-reduced MNIST at matched qubit budget.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
