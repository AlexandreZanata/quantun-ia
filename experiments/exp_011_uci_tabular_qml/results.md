# Results — EXP 011 (UCI Tabular QML)

**Run date:** 2026-06-17  
**Profile:** ci, 10 seeds
**Dataset:** breast_cancer (UCI), 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **perceptron** | 97.3% | [96.9%, 97.7%] |
| classical_matched_h4 | 96.9% | [96.4%, 97.3%] |
| quantum_angle | 91.3% | [88.2%, 93.4%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_angle vs classical_matched_h4 | -5.6 pp | 0.004 | -1.15 (large) | yes |
| perceptron vs quantum_angle | +6.0 pp | 0.004 | 1.22 (large) | yes |

## Verdict
**accepted** — primary comparison (quantum_angle vs classical_matched_h4) is Holm-significant with -1.15 (large).

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `ci`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Compare perceptron, parameter-matched MLP, and angle-encoding QNN on UCI tabular data. See `docs/baselines.md` for literature context.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
