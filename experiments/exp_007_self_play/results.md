# Results — EXP 007 (Self-Play — re-upload base)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, QuantumNetReupload (4q, 3 layers)

## Applicability gate
| Technique | Status | Base holdout | Threshold |
|-----------|--------|--------------|-----------|
| self_play | **applicable** | 59.3% | 55% |

## Holdout results
| Phase | Mean | 95% CI |
|-------|------|--------|
| self_play_base | 59.3% | [54.8%, 63.3%] |
| self_play_best | 60.4% | [57.1%, 63.6%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| self_play_best vs base | +1.1 pp | 1.0 | no |

## Conclusion
Self-play is now applicable with re-upload base but provides **no significant gain** (+1.1 pp). Checkpoint stabilization works; hard-example mining does not help on circles.

## Suggested ablation
- Higher hard_frac or more rounds with re-upload base.
