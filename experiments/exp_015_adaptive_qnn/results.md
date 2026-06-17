# Results — EXP 015 (Adaptive QNN)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** circles, noise=0.2, gradient-variance adaptive LR

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **quantum_6q_3l_adaptive** | 64.0% | [61.4%, 66.7%] |
| classical_matched_h14 | 62.2% | [58.8%, 65.3%] |
| quantum_6q_3l_fixed | 58.7% | [55.9%, 61.1%] |
| quantum_4q_2l_fixed | 56.2% | [52.7%, 60.4%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_6q_3l_adaptive vs quantum_6q_3l_fixed | +5.3 pp | 0.059 | 0.78 | no |
| quantum_6q_3l_adaptive vs quantum_4q_2l_fixed | +7.8 pp | 0.012 | 1.40 | yes |
| quantum_6q_3l_adaptive vs classical_matched_h14 | +1.8 pp | 0.576 | 0.35 | no |

## Conclusion
Primary test: adaptive vs fixed LR at 6q×3l. See `docs/literature_review.md`.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
