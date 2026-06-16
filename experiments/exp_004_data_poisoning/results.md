# Results — EXP 004 (Data Poisoning)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, clean holdout

## Holdout results
| Model | 0% poison | 30% poison |
|-------|-----------|------------|
| quantum_amplitude | **62.0%** | 58.7% |
| classical | 61.2% | 56.8% |
| quantum_angle | 51.3% | 52.5% |

## Paired Wilcoxon
| Comparison | Mean diff @0% | p @0% | @30% diff | p @30% |
|------------|---------------|-------|-----------|--------|
| classical vs amplitude | −0.8 pp | 0.488 | −1.9 pp | 0.125 |

## Conclusion
Amplitude competitive with classical at clean holdout; neither significantly wins. Angle stuck near chance. Classical slightly more robust at 30% poison (non-significant).
