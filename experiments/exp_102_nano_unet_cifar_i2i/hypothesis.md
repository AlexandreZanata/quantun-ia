# Hypothesis — EXP 102: NanoUNet CIFAR-10 I2I floor (Phase H-T1–H-T3)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase H (classical nano I2I floor)

## What I expect to happen

On `cifar10_v1` (train/val carved **before** normalize; official test held out), a compact
**NanoUNet** DDPM denoiser (~1–5M params) trained on GPU will produce samples whose
**FID (ResNet18 features)** against the val set is **≥ 20% lower** than a pure-noise
null baseline on the same feature extractor — establishing the classical I2I floor
quantum heads must beat.

## Why I expect this

- Phase G P0 (`exp_101`) delivered CIFAR packs + split indices on disk.
- DDPM on 32×32 RGB is a standard 4060-scale floor; noise null FID is a weak but
  honest comparator when a full teacher DDPM is not yet shipped.
- Roadmap H0 gate: recognizable samples + FID logged vs micro/null baseline.

## What would prove me wrong

- Model FID ≥ noise FID − 20% relative (no improvement)
- Collapse / OOM on RTX 4060 (8 GB)
- Val denoise MSE flat from epoch 1 (dead training)

## Metrics I will measure

- [x] Trainable parameter count
- [x] Final val noise-prediction MSE
- [x] FID-R18: model samples vs val
- [x] FID-R18: Gaussian noise images vs val (null)
- [x] Relative FID improvement = 1 − FID_model / FID_noise
- [x] LPIPS-proxy (VGG16 feature MSE) model vs val mean (logged honesty)
- [x] Wall-clock (s); device

## Success criteria

- **Primary (H0 / H-T3):** relative FID improvement ≥ **0.20** vs noise null on val
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- FID uses ImageNet-pretrained ResNet18 pools (not Inception-v3 FID); label as **FID-R18**
- LPIPS-proxy is VGG feature MSE, not the Kalantari/Zhang LPIPS package
- T2I TinyDiT (`exp_103`) and caption packs (G-T3) are out of scope for this run
- Sampling uses the same discrete schedule length as training (no DDIM acceleration)
