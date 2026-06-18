# Results — EXP 024 (QuantumNano-BC)

**Run date:** 2026-06-18  
**Profile:** publication, 30 seeds
**Dataset:** breast_cancer (UCI Wisconsin), full 569 samples, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **logistic_regression** | 97.9% | [97.6%, 98.2%] |
| perceptron | 97.6% | [97.1%, 98.0%] |
| hybrid_sandwich | 97.4% | [97.0%, 97.8%] |
| classical_matched_h5 | 97.0% | [96.7%, 97.3%] |
| xgboost_shallow | 96.2% | [95.8%, 96.6%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hybrid_sandwich vs logistic_regression | -0.5 pp | 0.003 | -0.72 (medium) | yes |
| hybrid_sandwich vs xgboost_shallow | +1.2 pp | 0.003 | 0.76 (medium) | yes |
| hybrid_sandwich vs classical_matched_h5 | +0.4 pp | 0.088 | 0.43 (small) | no |
| logistic_regression vs perceptron | +0.4 pp | 0.138 | 0.35 (small) | no |

## Verdict
**accepted** — primary comparison (hybrid_sandwich vs logistic_regression) is Holm-significant with -0.72 (medium).

## Power analysis
- Design: 30 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.51**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Flagship QuantumNano-BC benchmark: hybrid sandwich vs clinical baselines on full Wisconsin Breast Cancer. See model_cards/quantum_nano_bc.md.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
