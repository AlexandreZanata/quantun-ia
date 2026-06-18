# Results — EXP 031: Clinical Curriculum Ablation

**Run date:** 2026-06-18  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)
**Dataset:** breast_cancer (UCI Wisconsin), 30% holdout

## Holdout comparison

| Method | Mean |
|--------|------|
| curriculum_random | **96.84%** |
| curriculum_margin_batches | **97.02%** |

- Advantage: **+0.18 pp**
- Paired wins (curriculum > random): **4/10**
- Elapsed: **14.531s**

## Verdict
**accepted** — margin curriculum mean holdout vs epoch-matched random baseline.

## Limitations
- Single clinical dataset; not a deployment claim.
