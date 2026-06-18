# Results — EXP 027: Continuous Retrain Champion/Challenger Gate

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## Weekly cycles

| Week | Seed | Champion % | Challenger % | Δ pp | Decision |
|------|------|------------|--------------|------|----------|
| 1 | 123 | 96.49 | 96.49 | 0.00 | PROMOTED |
| 2 | 456 | 96.49 | 97.08 | 0.58 | PROMOTED |
| 3 | 789 | 97.08 | 97.66 | 0.58 | PROMOTED |
| 4 | 1024 | 97.66 | 96.49 | 1.17 | BLOCKED |

## Verdict
**accepted** — blocked cycles: 1; within 0.5 pp: 1/4; non-blocked: 3/4.

## Conclusion
Champion/challenger gate with `artifacts/champion/` symlink operates as designed for simulated weekly retrain on breast cancer holdout.

## Limitations
- Simulated weeks (sequential runs), not wall-clock cron.
- Single dataset; not a clinical deployment claim.
