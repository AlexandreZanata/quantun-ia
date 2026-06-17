# Hypothesis — EXP 018 (Feature Fusion)

**Date:** 2026-06-17  
**Author:** quantun-ia lab

## What I expect to happen

On a **phase-sensitive sequential task** (`sequential_phase`, 12×4), a **Transformer → QNN fusion**
model will beat both (a) flattened QNN and (b) PCA-compressed QNN by **≥ 3 percentage points**
mean holdout accuracy (10 seeds). It will be competitive with standalone `TransformerMini`.

## Why I expect this

Flattening destroys temporal order; PCA on flat windows cannot recover phase progression.
A transformer encoder pools sequence structure into a low-dimensional embedding suitable for
angle-encoded qubits — the fusion should combine temporal inductive bias with quantum expressivity.

## What would prove me wrong

- `transformer_qnn_fusion` ≤ `quantum_pca` or `quantum_flat` (fusion adds no value).
- `transformer_mini` dominates fusion by > 5 pp (quantum block hurts).
- All models near chance (~50%) — task too noisy or mis-specified.

## Metrics I will measure

- [x] Mean holdout accuracy per model (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: fusion vs PCA-QNN and fusion vs transformer baseline
- [x] Cohen's d on primary comparison

## Ablation plan

1. Halve `seq_len` (6) to reduce phase signal.
2. Disable quantum block (transformer encoder + classical head only).
3. Match qubit count to `pca_components` (4) across all quantum arms.
