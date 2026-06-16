# Results — EXP 008 (Data Re-uploading)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Parameter matching
| Model | n_params |
|-------|----------|
| quantum_reupload (4q, 3 layers) | 38 |
| classical_matched_h9 | 37 |

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| quantum_basic | 51.2% | [48.5%, 54.0%] |
| **quantum_reupload** | **58.6%** | [55.5%, 61.9%] |
| classical_matched_h9 | 56.9% | [54.0%, 59.3%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| reupload vs basic | +7.4 pp | 0.010 | **yes** |
| classical_matched vs reupload | −1.7 pp | 0.930 | no |

## Conclusion
Data re-uploading **significantly improves** QNN on circles (crosses the 55% learnability gate). Parameter-matched classical (h=9) is competitive but not significantly better than re-upload.

**Hypothesis partially confirmed:** re-upload beats basic QNN; classical still matches re-upload at equal param count.

## Suggested ablations
- Self-play on re-upload base model (exp_007 follow-up)
- Curriculum on re-upload base model (exp_005 follow-up)
- More layers vs more qubits at fixed param budget
