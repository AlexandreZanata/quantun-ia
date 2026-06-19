# Hypothesis — EXP 052: Quantum warm-start on HIGGS hybrid sandwich

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Training a **HybridSandwich** **classical-first for 70% of epochs**, then enabling the
QNN block for the final 30%, yields **≥ 0.5 pp ROC-AUC** over end-to-end hybrid on a
HIGGS 50k train subset — because classical layers find a good basin before variational
optimization.

## Why I expect this

- exp_002 showed architecture order matters; temporal schedule inverts static comparison.
- exp_015 GV-ALR helped plateau-prone QNN — warm-start may reduce early QNN gradient noise.
- exp_037 frozen-backbone pattern succeeded; warm-start is the sandwich analogue.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | warm-start val **ROC-AUC** ≥ end-to-end hybrid + **0.5 pp** (multi-seed mean) |
| Statistical | Wilcoxon p < 0.05 after Holm · **10 seeds** (publication) |
| Secondary | Training completes on RTX 4060 without OOM |

## Known limitations

- PennyLane QNN sim on CPU; HIGGS tabular uses batched classical pre/post on CUDA.
- CI uses 3 seeds and relaxed −2.0 pp gate — not a publication claim.
- Test split untouched; val ROC-AUC only for gate.
