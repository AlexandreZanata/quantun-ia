# Results — EXP 001 (Quantum vs Classical)

**Run date:** 2026-06-16 (re-upload follow-up)  
**Profile:** circles, n=500, noise=0.2, 10 seeds

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| classical_32 | **65.1%** | [62.5%, 67.5%] |
| quantum_reupload_4q_3l | **59.5%** | [55.7%, 62.8%] |
| classical_8 | 58.4% | [55.1%, 61.9%] |
| quantum_6q_3l | 56.1% | [53.4%, 58.8%] |
| quantum_4q_2l | 53.3% | [50.7%, 56.1%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| classical_32 vs quantum_4q_2l | +11.8 pp | 0.002 | **yes** |
| reupload vs basic 4q | +6.3 pp | 0.012 | **yes** |
| classical_32 vs reupload | +5.5 pp | 0.004 | **yes** |

## Conclusion
Data re-uploading closes much of the quantum gap but classical_32 still wins significantly. Basic QNN remains at chance; re-upload is the viable quantum baseline on circles.
