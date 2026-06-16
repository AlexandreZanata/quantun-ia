# Results — EXP 003 (Entanglement Effect)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, **re-upload QNN (4q, 3 layers)**

## Holdout results
| Entanglement | Mean | 95% CI |
|--------------|------|--------|
| none | **65.4%** | [63.9%, 67.1%] |
| chain_half | 64.4% | [61.9%, 67.1%] |
| ring | 62.0% | [58.8%, 65.3%] |
| chain | 60.9% | [57.6%, 64.3%] |

## Paired Wilcoxon vs `none` (Holm-Bonferroni)
| Comparison | Mean diff | p-value | p_holm | Significant |
|------------|-----------|---------|--------|-------------|
| chain vs none | −4.6 pp | 0.020 | 0.059 | no |
| chain_half vs none | −1.1 pp | 1.000 | 1.000 | no |
| ring vs none | −3.7 pp | 0.059 | 0.117 | no |

## Conclusion
With re-upload baseline, **no entanglement** tops all topologies — opposite of the original hypothesis. No pairwise comparison vs `none` survives Holm correction. Entanglement does not help on circles at this depth.
