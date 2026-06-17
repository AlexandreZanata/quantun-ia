# Hypothesis — EXP 013: Gaussian Augmentation Robustness

**Date:** 2026-06-17  
**Author:** Quantum ML Lab

## What I expect to happen
Adding Gaussian noise augmentation (sigma=0.15) to training circles data will improve QNN
holdout accuracy versus a non-augmented baseline on the same noisy circles task (noise=0.2).

## Why I expect this
Augmentation exposes the QNN to wider input perturbations during training, acting as
implicit regularization on the variational circuit.

## What would prove me wrong
- Augmented training does not improve holdout mean vs baseline
- Augmentation hurts holdout (over-regularization)

## Metrics I will measure
- [x] Holdout accuracy: baseline vs augmented (10 seeds)
- [x] Paired Wilcoxon comparison
