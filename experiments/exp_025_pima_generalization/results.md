# Results — EXP 025 (Pima Generalization)

**Run date:** 2026-06-18  
**Profile:** publication, 30 seeds
**Dataset:** pima_diabetes (OpenML id=37), full 768 samples, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **logistic_regression** | 77.2% | [76.4%, 78.0%] |
| perceptron | 76.5% | [75.6%, 77.3%] |
| xgboost_shallow | 76.2% | [75.4%, 77.1%] |
| classical_matched_h6 | 76.2% | [75.4%, 77.0%] |
| hybrid_sandwich | 76.2% | [75.3%, 77.1%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hybrid_sandwich vs logistic_regression | -1.0 pp | 0.012 | -0.62 (medium) | yes |
| hybrid_sandwich vs xgboost_shallow | -0.1 pp | 1.000 | -0.04 (negligible) | no |
| hybrid_sandwich vs classical_matched_h6 | -0.1 pp | 1.000 | -0.02 (negligible) | no |

## Verdict
**accepted (parity)** — `hybrid_sandwich` mean holdout is within the 2 pp parity threshold vs logistic regression (Δ=−1.0 pp). At n=30 seeds the gap is Holm-significant (p=0.012, medium d) but below the clinical parity band; generalization from exp_024 breast cancer holds on Pima.

## Power analysis
- Design: 30 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.51**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Pima Indians Diabetes generalization confirms **parity** with logistic regression (76.2% hybrid vs 77.2% logistic, Δ=−1.0 pp, 30 seeds). Hybrid QML matches strong tabular baselines on a second canonical benchmark; logistic regression remains the recommended deployable baseline for this task size.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
