# Results — EXP 021 (QML Backend Parity)

**Run date:** 2026-06-17  
**Profile:** publication, 10 seeds
**Dataset:** breast_cancer (UCI), angle QNN, 30% holdout

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **quantum_lightning** | 91.5% | [89.3%, 93.3%] |
| quantum_default | 91.1% | [88.0%, 93.3%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| quantum_default vs quantum_lightning | -0.4 pp | 0.844 | -0.07 (negligible) | no |

## Verdict
**accepted** — mean holdout difference −0.4 pp is within the 2 pp parity threshold; Wilcoxon not significant (p=0.844); Cohen's d negligible (−0.07). Backends are interchangeable for predictive accuracy on this task.

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
PennyLane default.qubit vs lightning.qubit holdout parity on breast cancer QNN. See `docs/compute_environment.md` for hardware profile.

## Limitations
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
