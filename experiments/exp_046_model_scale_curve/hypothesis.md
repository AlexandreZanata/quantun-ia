# Hypothesis — EXP 046: Model scale curve on HIGGS (805K)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

Training **five `LargeNanoMLP` width variants** (nano_s → nano_xxl) on the HIGGS v1 train
split (805K rows) will show **diminishing returns** past nano_l (~1.17M params):

- **nano_xl** (~3.5M) beats **nano_l** by **≥ 0.3 pp** validation ROC-AUC
- **nano_xxl** (~8M) either overfits nano_xl or **VRAM-fails** at batch 1024 on RTX 4060

## Why I expect this

- exp_032 established nano_l trains stably at batch 2048 on 805K HIGGS.
- Wider MLPs increase capacity but also VRAM and overfitting risk on fixed epochs.
- Roadmap VRAM envelope: nano_xl ~4.5 GB, nano_xxl ~6.5 GB (tight on 8 GB).

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Primary | Δ val ROC-AUC (nano_xl − nano_l) ≥ **0.3 pp** |
| Secondary | Peak VRAM logged per variant; train wall_time_s |
| nano_xxl | Document OOM or overfit — not required to pass |

## Variants

| Key | hidden1 | hidden2 | hidden3 | ~Params | batch |
|-----|---------|---------|---------|---------|-------|
| nano_s | 512 | 128 | 32 | ~50k | 4096 |
| nano_m | 1024 | 256 | 64 | ~300k | 2048 |
| nano_l | 2048 | 512 | 64 | ~1.17M | 2048 |
| nano_xl | 4096 | 1024 | 128 | ~3.5M | 2048 |
| nano_xxl | 4096 | 2048 | 256 | ~8M | 1024 |

## Known limitations

- Val-only selection; test split untouched.
- Single seed (42) for scale curve — multi-seed promotion deferred to Step 1.2.
- Classical MLP only — hybrid head ablation is Step 1.3.
