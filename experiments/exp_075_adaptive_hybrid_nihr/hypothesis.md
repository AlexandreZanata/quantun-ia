# Hypothesis — EXP 075: GV-ALR on frozen hybrid QNN head (NIHR C2 replication)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **exp_054** on **NIHR clinical tabular** (C2): **GV-ALR** on the trainable QNN head
with frozen `large_nano_mlp` backbone (exp_069) reaches the same val **PR-AUC** within **±0.3 pp**
using **≤ 70%** of fixed-LR head epochs.

## Why I expect this (or not)

- exp_054 on HIGGS was **accepted** (+0.01 pp AUC, 5/8 epochs, 0.59 wall-time ratio).
- NIHR is imbalanced clinical tabular — GV-ALR may or may not replicate the efficiency win.
- Completes the NIHR row for H-Q4 in the grand comparison matrix.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | \|adaptive PR-AUC − fixed PR-AUC\| ≤ **0.3 pp** |
| Efficiency | adaptive_epochs ≤ **70%** of fixed_epochs |
| Secondary | wall_time_s logged for both trainers |

## Known limitations

- PennyLane QNN sim on CPU; backbone on CUDA.
- Frozen C2 backbone from exp_069; val PR-AUC only.
- CI uses reduced row slice — not a publication claim.
