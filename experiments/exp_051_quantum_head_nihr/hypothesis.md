# Hypothesis — EXP 051: Quantum QNN head on frozen NIHR backbone

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

A **4-qubit re-upload QNN head** on a **frozen LargeNanoMLP backbone** (exp_044 checkpoint,
NIHR 13-feature) will match the classical sigmoid head within **−1.0 pp PR-AUC** on the
realistic-prevalence (~6.6%) NIHR holdout.

## Why I expect this

- exp_037 showed hybrid head training completes on frozen HIGGS backbone within −1.0 pp AUC.
- NIHR has realistic prevalence — PR-AUC is the honest primary metric (not accuracy).
- Head-only training (~300 params) limits barren-plateau risk vs full hybrid.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | Val **PR-AUC** (hybrid) ≥ classical_head − **1.0 pp** |
| Secondary | Training completes on RTX 4060 without OOM |
| Clinical ranking | Spearman ρ on exp_041 cases deferred (Synthea feature space ≠ NIHR) |

## Known limitations

- PennyLane QNN sim on CPU; backbone on CUDA.
- Clinical case validation requires NIHR-native profiles (exp_060).
- Test split untouched; val PR-AUC only for gate.
