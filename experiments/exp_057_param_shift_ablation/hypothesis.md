# Hypothesis — EXP 057: Parameter-shift vs autograd on deep re-upload QNN

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane QNN on CPU)

## What I expect to happen

On **4-qubit × 3-layer** re-upload QNN trained on **breast cancer**, **parameter-shift** holdout
accuracy stays within **1 pp** of **autograd (backprop)** while showing **≥ 2× lower mean gradient
variance** (exp_006 metric) across **10 seeds**.

## Why I expect this

- exp_006 confirmed barren-plateau gradient decay; parameter-shift is the canonical exact-gradient
  rule for variational circuits.
- Deep re-upload (3 layers) is where `circuit_utils` already recommends parameter-shift for
  diagnostics — this tests whether it helps **training stability** at matched accuracy.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Holdout accuracy | \|autograd − param-shift\| ≤ **1.0 pp** (mean over seeds) |
| Gradient variance | autograd_var / param_shift_var ≥ **2.0** |
| Secondary | Training completes on RTX 4060 without OOM |

## Known limitations

- Parameter-shift requires batch_size=1 — slower than autograd full-batch.
- PennyLane QNN sim on CPU; classical pre/post layers may use CUDA when enabled.
- CI uses row caps and relaxed gates — not a publication claim.
