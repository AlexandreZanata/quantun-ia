# Results — EXP 005 (Curriculum Quantum — re-upload base)

**Run date:** 2026-06-16  
**Profile:** circles, n=500, noise=0.2, 10 seeds, QuantumNetReupload (4q, 3 layers)

## Applicability gate
| Technique | Status | Base holdout | Threshold |
|-----------|--------|--------------|-----------|
| curriculum | **applicable** | 60.1% | 55% |

## Holdout results
| Method | Mean | 95% CI |
|--------|------|--------|
| curriculum_random | **60.1%** | [57.0%, 63.3%] |
| curriculum_margin_batches | 54.5% | [50.7%, 58.3%] |

## Paired Wilcoxon
| Comparison | Mean diff | p-value | Significant |
|------------|-----------|---------|-------------|
| margin_batches vs random | −5.7 pp | 0.084 | no |

## Conclusion
With a learnable re-upload base, curriculum runs but **margin_batches underperforms random** (non-significant trend). Honest negative for curriculum on this task/architecture.

## Suggested ablation
- Fewer curriculum stages or longer refine phase on full data.
