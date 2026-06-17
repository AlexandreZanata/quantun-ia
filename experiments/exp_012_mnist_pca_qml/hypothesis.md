# Hypothesis — EXP 012: MNIST PCA QML Encoding Comparison

**Date:** 2026-06-17  
**Author:** Quantum ML Lab

## What I expect to happen
On PCA-reduced MNIST digits (0 vs 1, 8 components, 4 qubits), amplitude encoding will
outperform angle encoding because the projected features span a higher-dimensional Hilbert space.

## Why I expect this
Amplitude encoding uses 2^n amplitudes; angle encoding uses only n angles per layer.
Binary digit discrimination may need the extra expressivity.

## What would prove me wrong
- Angle encoding matches or beats amplitude within 2 pp across seeds
- Both encodings stay near chance (~50%) after PCA compression

## Metrics I will measure
- [x] Holdout accuracy per encoding (10 seeds, bootstrap CI)
- [x] Paired Wilcoxon: amplitude vs angle
- [x] PCA variance explained (logged in protocol)
