# Results — EXP 026: Real Application E2E (API = CLI on GPU)

**Date:** 2026-06-18  
**Profile:** `ci` (5 seeds, 12 epochs)  
**Hardware:** NVIDIA RTX 4060 Laptop GPU, `QML_DEVICE=cuda`  
**Verdict:** **Accepted** — max |Δ| = 0.00 pp (threshold 0.5 pp)

## Summary

Async REST API training jobs with `device=cuda` produce **identical** holdout accuracy to the
synchronous CLI path (`train_nanomodel.execute`) for `hybrid_sandwich` × Wisconsin Breast Cancer
across all 5 CI seeds.

## Per-seed results

| Seed | CLI holdout | API holdout | Δ pp | CLI time (s) | API time (s) |
|------|-------------|-------------|------|--------------|--------------|
| 42 | 94.15% | 94.15% | 0.00 | 0.2 | 0.3 |
| 123 | 97.66% | 97.66% | 0.00 | 0.2 | 0.3 |
| 456 | 64.91% | 64.91% | 0.00 | 0.2 | 0.5 |
| 789 | 98.25% | 98.25% | 0.00 | 0.2 | 0.5 |
| 1024 | 94.74% | 94.74% | 0.00 | 0.2 | 0.5 |

**Mean |Δ|:** 0.000 pp  
**Max |Δ|:** 0.000 pp

## Bug fixed during run

Initial run showed large CLI/API drift (up to 31.6 pp on seed 42). Root cause: model weights were
initialized **before** `set_global_seed()` in `train_nanomodel.execute`. Fix: call
`set_global_seed(seed)` before `build_model()`. After fix, parity is exact.

## Conclusion

The real application stack (async API + CUDA worker + SQLite job queue) is **scientifically
equivalent** to the direct CLI training path. Phase B exit criteria met.

## Reproduce

```bash
QML_DEVICE=cuda MLFLOW_DISABLE=1 make exp-026
```
