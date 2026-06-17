# Results — EXP 016 (Hybrid NAS)

**Run date:** 2026-06-17  
**Profile:** ci, 10 seeds
**Dataset:** circles, noise=0.2, Optuna over hybrid layouts

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **quantum_first** | 65.5% | [62.9%, 67.9%] |
| nas_best | 64.7% | [61.1%, 67.9%] |
| hybrid_sandwich | 62.1% | [59.1%, 65.0%] |
| classical_first | 62.1% | [59.1%, 65.0%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| nas_best vs hybrid_sandwich | +2.6 pp | 0.059 | 0.90 (large) | no |
| nas_best vs quantum_first | -0.8 pp | 0.477 | -0.31 (small) | no |
| nas_best vs classical_first | +2.6 pp | 0.059 | 0.90 (large) | no |

## Verdict
**rejected** — primary comparison (nas_best vs hybrid_sandwich) not significant after Holm correction (0.90 (large)).

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `ci`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
NAS-best hybrid vs fixed EXP 002 baselines. See `docs/literature_review.md` §5.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
