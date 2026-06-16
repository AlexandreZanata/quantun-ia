# Hypothesis — EXP 001: Quantum vs Classical

**Date:** 2026-06-16  
**Author:** Quantum ML Lab

## What I expect to happen
On circles (noise=0.2, n=500), a classical MLP with 32 hidden units will outperform
variational QNNs because the decision boundary is non-linear and classical networks
have more effective capacity per parameter. Basic angle-encoding QNNs will stay near
chance (~50%). Data re-uploading QNN should improve over basic QNN but still trail
classical_32.

## Why I expect this
EXP 001 is the primary benchmark. Shallow QNNs with single data upload struggle on
concentric circles; classical MLPs are a strong baseline. Re-uploading (exp_008 finding)
adds expressivity without more qubits.

## What would prove me wrong
- quantum_4q_2l holdout mean ≥ classical_32 (Wilcoxon p < 0.05)
- quantum_reupload_4q_3l beats classical_32 significantly
- All models within 2 pp of each other (no clear winner)

## Metrics I will measure
- [x] Holdout accuracy per model (10 seeds, bootstrap 95% CI)
- [x] Paired Wilcoxon: classical_32 vs quantum_4q_2l, reupload vs basic, classical vs reupload
- [x] Training time and n_params per model
