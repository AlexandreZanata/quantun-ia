# Results — EXP 024 (QuantumNano-BC)

**Run date:** 2026-06-18  
**Profile:** publication, 30 seeds
**Dataset:** breast_cancer (UCI Wisconsin), full 569 samples, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **logistic_regression** | 97.9% | [97.6%, 98.2%] |
| perceptron | 97.6% | [97.1%, 98.0%] |
| hybrid_sandwich | 97.4% | [97.0%, 97.7%] |
| classical_matched_h5 | 97.0% | [96.6%, 97.3%] |
| xgboost_shallow | 96.2% | [95.8%, 96.6%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hybrid_sandwich vs logistic_regression | -0.5 pp | 0.003 | -0.68 (medium) | yes |
| hybrid_sandwich vs xgboost_shallow | +1.2 pp | 0.003 | 0.75 (medium) | yes |
| hybrid_sandwich vs classical_matched_h5 | +0.4 pp | 0.057 | 0.43 (small) | no |
| logistic_regression vs perceptron | +0.3 pp | 0.177 | 0.34 (small) | no |

## Verdict
**accepted (parity)** — `hybrid_sandwich` mean holdout is within the 2 pp parity threshold vs logistic regression (Δ=−0.5 pp). The advantage claim (≥3 pp) is **rejected**. At n=30 seeds the −0.5 pp gap is Holm-significant (p=0.003, medium d) but clinically negligible for this benchmark scope.

## Power analysis
- Design: 30 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.51**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Flagship QuantumNano-BC achieves **parity** with logistic regression on full Wisconsin Breast Cancer (97.4% vs 97.9% mean holdout). Classical logistic regression remains the recommended shippable baseline; hybrid QML is validated as a reproducible experimental variant. See `model_cards/quantum_nano_bc.md`.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
