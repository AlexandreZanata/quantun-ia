# Results — EXP 013 (Augmentation Robustness)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** noisy circles, Gaussian augmentation

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **augmented** | 58.7% | [53.9%, 63.3%] |
| baseline | 56.7% | [53.5%, 60.2%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| augmented vs baseline | +2.0 pp | 0.506 | 0.23 | no |

## Conclusion
Gaussian augmentation vs baseline QNN on noisy circles.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
