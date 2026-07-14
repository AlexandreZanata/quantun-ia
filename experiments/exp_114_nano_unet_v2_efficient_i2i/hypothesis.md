# Hypothesis — EXP 114: NanoUNet-v2 efficient I2I (Phase M / M0)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v4 · Phase M (efficient classical nano I2I)

## What I expect to happen

On Phase L packs (`tiny_imagenet_v1` primary at **64×64**, `stl10_v1` resized 96→64 for side card), a **NanoUNet-v2** DDPM with an explicit param budget will:

1. Beat a pure-noise null by **≥ 20% relative FID-R18** on the same val split (not near-noise).  
2. Either improve absolute FID vs exp_102 CIFAR floor (**≤ 0.9 × 153.93 ≈ 138.5**) **or** ship a clear **efficiency card** (params, peak VRAM, imgs/s) with recognizable samples.

That establishes the Cycle-v4 efficient I2I floor before TinyDiT-latent T2I (`exp_115`) and latent trainers (Phase N).

## Why I expect this

- Phase L delivered split-before-resize Tiny-IN / STL-10 on disk (`exp_113`).  
- exp_102 proved NanoUNet DDPM at 32×32 (FID-R18 153.93, rel 0.764).  
- NanoUNet-v2 adds a second downsample stage for 64×64 while keeping base channels modest for 8 GB VRAM.

## What would prove me wrong

- Relative FID improvement < 0.20 vs noise (still near-noise)  
- OOM on RTX 4060 or dead training (flat loss from epoch 1)  
- Efficiency card missing when absolute FID gate fails

## Metrics I will measure

- [x] Trainable parameter count  
- [x] Val denoise MSE  
- [x] FID-R18 model vs val · noise vs val · relative improvement  
- [x] LPIPS-proxy  
- [x] Peak CUDA memory (MB) · train imgs/s · wall-clock  
- [x] Device / pack / img_size

## Success criteria

- **Primary (M0):** rel FID ≥ **0.20** vs noise **and** (FID ≤ **138.5** **or** efficiency card complete)  
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Absolute FID vs exp_102 mixes 32×32 CIFAR with 64×64 Tiny-IN — relative-to-noise is the honesty gate; absolute is secondary.  
- FID-R18 / LPIPS-proxy same caveats as exp_102.  
- T2I repair (`exp_115`) out of scope.
