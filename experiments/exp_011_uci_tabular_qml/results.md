# Results — EXP 011 (UCI Tabular QML)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** breast_cancer (UCI), 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **perceptron** | 97.2% | [96.8%, 97.6%] |
| classical_matched_h4 | 97.1% | [96.9%, 97.4%] |
| quantum_angle | 91.5% | [88.4%, 93.6%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_angle vs classical_matched_h4 | -5.7 pp | 0.004 | -1.17 | yes |
| perceptron vs quantum_angle | +5.7 pp | 0.004 | 1.13 | yes |

## Conclusion
Compare perceptron, parameter-matched MLP, and angle-encoding QNN on UCI tabular data. See `docs/baselines.md` for literature context.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
