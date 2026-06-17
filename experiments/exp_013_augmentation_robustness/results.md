# Results — EXP 013 (Augmentation Robustness)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** noisy circles, Gaussian augmentation

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **augmented** | 58.7% | [53.9%, 63.3%] |
| baseline | 56.8% | [53.5%, 60.2%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| augmented vs baseline | +1.9 pp | 0.510 | 0.22 (small) | no |

## Verdict
**inconclusive** — |d|=0.22 below MDE=0.89 for n=10 seeds (underpowered).

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Gaussian augmentation vs baseline QNN on noisy circles.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
