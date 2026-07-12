# Hypothesis — EXP 065: GV-ALR on frozen hybrid QNN head (ACYD C4 / H-Q4)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Replicating **H-Q4** (exp_054 / exp_075 protocol) on **ACYD Brazil soybean** (C4): **GV-ALR** on the
trainable QNN head with frozen `large_nano_mlp` backbone (exp_060) reaches the same val **ROC-AUC**
within **±0.3 pp** using **≤ 70%** of fixed-LR head epochs.

## Why I expect this (or not)

- exp_054 on HIGGS and exp_075 on NIHR were **accepted** on efficiency gates.
- ACYD is agro-climate tabular with temporal split — GV-ALR may or may not replicate the efficiency win.
- Completes the ACYD row for H-Q4 in the grand comparison matrix.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | \|adaptive ROC-AUC − fixed ROC-AUC\| ≤ **0.3 pp** |
| Efficiency | adaptive_epochs ≤ **70%** of fixed_epochs |
| Secondary | wall_time_s logged for both trainers |

## Known limitations

- PennyLane QNN sim on CPU; backbone on CUDA.
- Frozen C4 backbone from exp_060; val ROC-AUC only.
- CI uses reduced row slice — not a publication claim.
