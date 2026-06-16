# Results — EXP 008 (Data Re-uploading)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Holdout results
| Model | Mean | 95% CI | Params |
|-------|------|--------|--------|
| quantum_reupload | **58.1%** | [54.3%, 62.0%] | 38 |
| classical_matched_h9 | 57.5% | [54.4%, 60.3%] | 37 |
| quantum_basic | 52.5% | [50.2%, 55.1%] | 22 |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| reupload vs basic | +5.6 pp | 0.012 | **yes** |
| classical_matched vs reupload | −0.6 pp | 0.791 | no |

## Conclusion
Re-upload significantly beats basic QNN. Parameter-matched classical equivalent at equal param budget.
