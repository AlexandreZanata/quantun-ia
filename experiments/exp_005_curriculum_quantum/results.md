# Results — EXP 005 (Curriculum — re-upload base)

**Run date:** 2026-06-16 (epoch-matched baseline: 60 epochs)  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Applicability gate: **applicable** (random mean 60.2%)

## Holdout results
| Method | Mean | 95% CI |
|--------|------|--------|
| curriculum_random | **60.2%** | [57.7%, 62.8%] |
| curriculum_margin_batches | 55.3% | [50.9%, 59.4%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| margin_batches vs random | −4.9 pp | 0.125 | no |

## Conclusion
After epoch-matching, curriculum still trends worse than random. Honest negative — margin-based staging hurts on circles.
