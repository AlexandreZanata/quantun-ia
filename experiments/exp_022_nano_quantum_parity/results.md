# Results — EXP 022 (Nano Quantum Parity)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Protocol:** parameter-matched classical MLP vs quantum nanomodel (|Δparams| ≤ 10)

## Holdout results
| Dataset | Quantum model | Classical baseline | Quantum mean | Classical mean | Δ (pp) |
|---------|---------------|-------------------|--------------|----------------|--------|
| breast_cancer | hybrid_sandwich | classical_matched_h5 | 97.2% | 96.7% | +0.5 |
| wine_binary | hybrid_sandwich | classical_matched_h6 | 98.5% | 97.2% | +1.3 |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| hybrid_sandwich vs classical_matched_h5 (breast_cancer) | +0.5 pp | 0.355 | 0.36 (small) | no |
| hybrid_sandwich vs classical_matched_h6 (wine_binary) | +1.3 pp | 0.125 | 0.71 (medium) | no |

## Verdict
**inconclusive** — positive mean gaps on some datasets but Wilcoxon not Holm-significant at α=0.05 (underpowered or high variance).

## Power analysis
- Design: 10 paired holdout accuracies per dataset (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Primary claim threshold: ≥2 pp mean difference + Holm-significant Wilcoxon.

## Conclusion
Tests whether hybrid sandwich with data re-upload beats a parameter-matched classical MLP on UCI tabular at equal trainable-parameter budget.

## Limitations
- Single holdout split per seed; no nested CV.
- Classical baseline hidden size chosen by parameter count, not architecture search.
- Results specific to `nano_parity_bench.yaml` seeds and epochs.
