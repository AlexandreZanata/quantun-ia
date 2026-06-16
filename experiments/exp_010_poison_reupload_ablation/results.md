# Results — EXP 010 (Poison Re-upload Ablation)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, clean holdout eval

## Holdout results
| Variant | 0% poison | 30% poison | Drop |
|---------|-----------|------------|------|
| reupload_3l | **58.3%** | 53.7% | 4.5 pp |
| reupload_lr_low | 55.4% | 51.6% | 3.8 pp |
| reupload_2l | 54.4% | 52.9% | **1.5 pp** |

## Paired Wilcoxon vs reupload_3l (Holm-Bonferroni)
| Comparison | Mean diff @0% | p_holm | @30% diff | p_holm |
|------------|---------------|--------|-----------|--------|
| reupload_2l | −3.9 pp | 0.172 | −0.8 pp | 0.883 |
| reupload_lr_low | −2.9 pp | 0.826 | −2.1 pp | 0.883 |

## Conclusion
**3 layers still best on clean holdout** — fewer layers or lower LR do not improve accuracy. However, `reupload_2l` is the most poison-robust (smallest drop at 30%). Trade-off: capacity vs robustness under label noise.
