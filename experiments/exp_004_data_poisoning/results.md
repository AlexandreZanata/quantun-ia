# Results — EXP 004 (Data Poisoning)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, clean holdout, **re-upload vs amplitude**

## Holdout results
| Model | 0% poison | 30% poison |
|-------|-----------|------------|
| quantum_amplitude | **61.5%** | 57.1% |
| classical | 57.5% | 57.5% |
| quantum_reupload | 56.2% | 56.2% |

## Paired Wilcoxon (Holm-Bonferroni)
| Comparison | Mean diff | p-value | p_holm | Significant |
|------------|-----------|---------|--------|-------------|
| classical vs reupload @0% | −4.2 pp | 0.219 | 0.656 | no |
| classical vs reupload @30% | +1.3 pp | 0.434 | 0.723 | no |
| amplitude vs reupload @0% | +2.9 pp | 0.361 | 0.723 | no |

## Conclusion
Amplitude encoding leads on clean holdout; re-upload underperforms amplitude here (unlike exp_008 standalone). No significant differences after Holm correction. Classical and re-upload show flat degradation at 30% poison (train labels flipped, holdout clean).
