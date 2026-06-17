# Results — EXP 018 (Feature Fusion)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** sequential_phase 12×4, PCA-insufficient temporal task

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **transformer_mini** | 96.9% | [96.1%, 97.7%] |
| quantum_flat | 95.7% | [94.5%, 96.8%] |
| transformer_qnn_fusion | 95.1% | [93.8%, 96.3%] |
| quantum_pca | 59.0% | [55.7%, 62.3%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| transformer_qnn_fusion vs quantum_pca | +36.1 pp | 0.006 | 6.13 (large) | yes |
| transformer_qnn_fusion vs transformer_mini | -1.8 pp | 0.023 | -1.17 (large) | yes |
| transformer_qnn_fusion vs quantum_flat | -0.5 pp | 0.441 | -0.19 (negligible) | no |

## Verdict
**accepted** — primary comparison (transformer_qnn_fusion vs quantum_pca) is Holm-significant with 6.13 (large).

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Transformer → QNN fusion vs PCA-QNN and flat QNN on phase-sensitive sequences.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
