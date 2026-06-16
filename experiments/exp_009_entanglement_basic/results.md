# Results — EXP 009 (Entanglement Basic Ablation)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, **basic QNN (no re-upload, 4q, 2 layers)**

## Holdout results
| Entanglement | Mean | 95% CI |
|--------------|------|--------|
| chain_half | **61.3%** | [58.7%, 63.9%] |
| none | 60.5% | [57.5%, 63.3%] |
| chain | 57.8% | [54.0%, 61.7%] |
| ring | 56.4% | [53.9%, 59.1%] |

## Paired Wilcoxon vs `none` (Holm-Bonferroni)
| Comparison | Mean diff | p-value | p_holm | Significant |
|------------|-----------|---------|--------|-------------|
| chain_half vs none | +0.8 pp | 0.711 | 0.711 | no |
| chain vs none | −2.7 pp | 0.301 | 0.602 | no |
| ring vs none | −4.1 pp | 0.098 | 0.293 | no |

## Conclusion
Without re-upload, topologies cluster near 56–61% with **no significant entanglement effect**. The exp_003 reversal (`none` > chain) appears tied to re-upload expressivity, not entanglement alone.
