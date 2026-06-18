# Results — EXP 025 (Pima Generalization)

**Run date:** 2026-06-18  
**Profile:** publication, 30 seeds
**Dataset:** pima_diabetes (OpenML id=37), full 768 samples, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **logistic_regression** | 77.2% | [76.4%, 78.0%] |
| perceptron | 76.5% | [75.6%, 77.3%] |
| classical_matched_h6 | 76.3% | [75.7%, 77.1%] |
| xgboost_shallow | 76.2% | [75.4%, 77.1%] |
| hybrid_sandwich | 76.2% | [75.3%, 77.1%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hybrid_sandwich vs logistic_regression | -1.0 pp | 0.017 | -0.59 (medium) | yes |
| hybrid_sandwich vs xgboost_shallow | -0.0 pp | 0.810 | -0.02 (negligible) | no |
| hybrid_sandwich vs classical_matched_h6 | -0.2 pp | 0.810 | -0.06 (negligible) | no |

## Verdict
**accepted** — primary comparison (hybrid_sandwich vs logistic_regression) is Holm-significant with -0.59 (medium).

## Power analysis
- Design: 30 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.51**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Generalization benchmark: hybrid sandwich vs clinical baselines on Pima Indians Diabetes. Compare verdict to exp_024 breast cancer parity.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
