# Hypothesis — EXP 055: Depolarizing noise regularization on GoBug hybrid QNN

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

**Depolarizing noise** (p=0.01–0.05) during QNN forward on **GoBug file-level** data improves
**PR-AUC on the temporal test split** (latest sha-order holdout) by **≥ 0.5 pp** vs a noiseless
HybridSandwich — analogous to dropout but respecting quantum channel semantics.

## Why I expect this

- exp_004 studied poisoning on encoding; channel noise is an unexplored regularizer on real SDP data.
- GoBug temporal split (exp_045) exposes drift; noise may improve generalization to later commits.
- Small hybrid head (~145 params) is prone to overfit on ~31% prevalence file labels.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | noisy test **PR-AUC** ≥ noiseless + **0.5 pp** |
| Secondary | Training completes on RTX 4060 without OOM |
| Baselines | noiseless hybrid · logistic (logged, not gating) |

## Known limitations

- PennyLane QNN sim on CPU; GoBug tabular pre/post on CUDA.
- Temporal split is sha-order proxy, not wall-clock timestamps.
- CI uses row caps and relaxed gate — not a publication claim.
