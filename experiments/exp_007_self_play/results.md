# Results — EXP 007 (Self-Play — re-upload base)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Applicability gate: **applicable** (base mean 60.8%)

## Holdout results
| Phase | Mean | 95% CI |
|-------|------|--------|
| self_play_best | **61.3%** | [58.4%, 64.1%] |
| self_play_base | 60.8% | [57.4%, 63.9%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| best vs base | +0.5 pp | 0.500 | no |

## Conclusion
Self-play applicable but provides no significant gain. Checkpoint stabilization works; hard-example mining neutral on circles.
