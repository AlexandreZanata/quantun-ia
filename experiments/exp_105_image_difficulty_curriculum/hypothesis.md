# Hypothesis — EXP 105: Image difficulty curriculum on CIFAR (Phase I / H-I2)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase I (transfer winning trainers)

## What I expect to happen

On `cifar10_v1`, training NanoUNet DDPM with an **easy→hard sharpness curriculum**
(Laplacian variance: blurry/smooth first, sharp/detailed last) with cumulative
stages + full-data refine will beat the same architecture under **random-order
staged training** by ≥ **5% relative FID-R18**, matched epoch budget.

## Why I expect this

- Cycle v2 SPEI curriculum (`exp_097`) won +0.83 pp with easy→hard staging.
- Roadmap H-I2 ports that pattern to pixels via blur/sharpness (I2I; no captions).
- Soft denoise distill (`exp_104`) failed teacher proximity — curriculum is the
  next I-T2 arm for I0 gate recovery.

## What would prove me wrong

- Curriculum FID ≥ random × 0.95 (no ≥5% relative win)
- Collapse / OOM on RTX 4060
- Sharpness scores uninformative (curriculum ≈ random within noise)

## Metrics I will measure

- [x] FID-R18 random-order staged
- [x] FID-R18 sharpness curriculum
- [x] Relative FID win: `1 − FID_curr / FID_random`
- [x] Trainable params; wall-clock; device

## Success criteria

- **Primary (H-I2 / I-T2):** relative FID win ≥ **0.05** vs random staged
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Difficulty = grayscale Laplacian variance (not full LPIPS / caption length)
- Cumulative stages + refine match total epochs; no SPEI agro features
- GV-ALR (`exp_105b`) deferred
