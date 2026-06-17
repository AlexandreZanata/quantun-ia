# Results — EXP 014 (Sequence Baselines)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** sequential_binary synthetic

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **transformer_mini** | 97.1% | [96.3%, 98.1%] |
| rnn_mini | 94.7% | [93.7%, 95.7%] |
| quantum_flat | 92.3% | [89.8%, 94.5%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| transformer_mini vs rnn_mini | +2.5 pp | 0.023 | 1.07 | yes |
| quantum_flat vs rnn_mini | -2.4 pp | 0.273 | -0.47 | no |

## Conclusion
RNN-mini and Transformer-mini vs flattened QNN on sequential task.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
