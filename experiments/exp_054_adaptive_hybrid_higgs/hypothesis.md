# Hypothesis — EXP 054: GV-ALR on frozen hybrid QNN head (HIGGS)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

**GV-ALR** (exp_015) applied only to the **trainable QNN head** on a frozen LargeNanoMLP
backbone (exp_037 setup) reaches the same val ROC-AUC within **±0.3 pp** using **≤ 70%**
of fixed-LR head epochs — a compute-efficiency win on CPU-simulated QNN.

## Why I expect this

- exp_015 showed GV-ALR helps plateau-prone QNN on circles; head-only training is shallow.
- exp_037 fixed-LR head converges in few epochs — adaptive scaling may reach parity faster.
- Barren plateau risk is low (4 qubits × 2 layers, frozen backbone).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | \|adaptive AUC − fixed AUC\| ≤ **0.3 pp** |
| Efficiency | adaptive_epochs ≤ **70%** of fixed_epochs |
| Secondary | wall_time_s logged for both trainers |

## Known limitations

- PennyLane QNN sim on CPU; backbone on CUDA.
- CI uses 8K train slice — not full 805K HIGGS train.
- Test split untouched; val AUC only for gate.
