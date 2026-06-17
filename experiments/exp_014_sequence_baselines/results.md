# Results — EXP 014 (Sequence Baselines)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** sequential_binary synthetic

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **transformer_mini** | 96.7% | [95.2%, 98.0%] |
| rnn_mini | 94.7% | [93.9%, 95.7%] |
| quantum_flat | 92.5% | [89.9%, 94.8%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| transformer_mini vs rnn_mini | +2.0 pp | 0.125 | 0.71 (medium) | no |
| quantum_flat vs rnn_mini | -2.2 pp | 0.379 | -0.42 (small) | no |

## Verdict
**inconclusive** — |d|=0.71 below MDE=0.89 for n=10 seeds (underpowered).

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
RNN-mini and Transformer-mini vs flattened QNN on sequential task.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
