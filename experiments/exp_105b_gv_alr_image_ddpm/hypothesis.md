# Hypothesis — EXP 105b: GV-ALR on NanoUNet DDPM (Phase I / H-I3)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase I (transfer winning trainers)

## What I expect to happen

On `cifar10_v1`, **GV-ALR** (gradient-variance adaptive LR on denoise MSE) training a
NanoUNet DDPM for ≤ **70%** of a fixed-LR epoch budget will match fixed-LR **FID-R18
within ±3% relative** — the same efficiency gate as exp_054 / exp_065 / exp_075.

## Why I expect this

- Cycle-1 GV-ALR won epoch-efficiency on frozen hybrid heads (HIGGS / ACYD / NIHR).
- Roadmap H-I3 / I-T3 wires the same `AdaptiveLRConfig` into the image DDPM trainer.
- Distill (H-I1) and sharpness curriculum (H-I2) failed; efficiency transfer is the
  remaining Phase I accept path for I0.

## What would prove me wrong

- `|FID_gvalr − FID_fixed| / FID_fixed > 0.03`
- Adaptive epochs > 0.70 × fixed epochs
- Collapse / OOM on RTX 4060

## Metrics I will measure

- [x] Fixed-LR FID-R18; epochs; wall-clock
- [x] GV-ALR FID-R18; epochs; wall-clock
- [x] Relative FID delta vs fixed
- [x] Epoch fraction adaptive / fixed
- [x] Trainable params; device

## Success criteria

- **Primary (H-I3):** FID within ±3% relative **and** epochs ≤ 70% of fixed
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- GV-ALR adapts on denoise MSE grad variance (not BCE / QNN head)
- FID-R18 proxy, not Inception FID
- Phase I I0 still open if this rejects (prefer Phase J latent path)
