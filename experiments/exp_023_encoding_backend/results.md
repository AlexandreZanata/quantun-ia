# Results — EXP 023 (Encoding × Backend)

**Run date:** 2026-06-18  
**Profile:** publication, 10 seeds
**Dataset:** MNIST 0 vs 1, PCA-8, angle vs amplitude × default/lightning backends

## Holdout results
| Model | Mean | 95% CI |
|-------|------|--------|
| **amplitude_default** | 99.9% | [99.7%, 100.0%] |
| angle_default | 93.9% | [90.9%, 96.8%] |
| angle_lightning | 93.3% | [89.0%, 97.4%] |
| amplitude_lightning | nan% | [nan%, nan%] |

## Paired Wilcoxon (Holm-Bonferroni where batched)
| Comparison | Mean diff | p-value | Cohen's d | Significant |
|------------|-----------|---------|-----------|-------------|
| angle_default vs angle_lightning | +0.6 pp | 1.000 | 0.06 (negligible) | no |
| angle_default vs amplitude_default | -5.9 pp | 0.012 | -1.14 (large) | yes |

## Verdict
**partial** — angle encoding backend parity **accepted** (+0.6 pp, p=1.000, negligible d). Amplitude vs angle on `default.qubit` **significant** (−5.9 pp, Holm p=0.012). `amplitude_lightning` failed all 10 seeds (Mottonen NaN) — backend×encoding interaction for amplitude **not testable** on lightning.

## Power analysis
- Design: 10 paired holdout accuracies per model (profile `publication`).
- Minimum detectable |Cohen's d| at α=0.05, power=0.80: **0.89**.
- Run `make power-analysis` or `python scripts/power_analysis.py --table` for other seed counts.

## Conclusion
Angle encoding shows backend parity (consistent with exp_021). Amplitude encoding on `default.qubit` strongly outperforms angle on this PCA-MNIST task (replicating exp_012 direction). Lightning backend cannot complete amplitude encoding in this environment — document per-cell coverage before claiming full 2×2 factorial replication.

## Limitations
- `amplitude_lightning`: 0/10 seeds completed (`state_vector` norm NaN on Mottonen decomposition).
- Single holdout split protocol; no nested CV.
- Results specific to the profile and seed list in `config/experiments.yaml`.
- Compare absolute metrics to literature only with protocol alignment (`docs/baselines.md`).
