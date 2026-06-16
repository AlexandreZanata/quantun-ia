# Results — EXP 007 (Self-Play — re-upload base)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Applicability gate: **applicable** (base mean 57.4%)

## Holdout results
| Phase | Mean | 95% CI |
|-------|------|--------|
| self_play_best | **58.5%** | [56.7%, 60.4%] |
| self_play_base | 57.4% | [54.3%, 60.1%] |

## Paired Wilcoxon (Holm-Bonferroni)
| Comparison | Mean diff | p-value | p_holm | Significant |
|------------|-----------|---------|--------|-------------|
| best vs base | +1.1 pp | 1.000 | 1.000 | no |

## Conclusion
Self-play applicable (base 57.4%) but provides no significant gain with Holm correction. Checkpoint stabilization works; hard-example mining neutral on circles.

---

## Publication large (n=1000)

| Phase | Mean | 95% CI |
|-------|------|--------|
| self_play_best | 62.9% | [59.3%, 66.1%] |
| self_play_base | 62.9% | [59.3%, 66.1%] |

Zero gain at larger sample size; see `docs/publication_large_summary.md`.
