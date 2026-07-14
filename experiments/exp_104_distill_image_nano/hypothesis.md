# Hypothesis — EXP 104: Teacher→NanoUNet image distill (Phase I / H-I1)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase I (transfer winning trainers)

## What I expect to happen

On `cifar10_v1`, a **student NanoUNet** trained with soft denoise targets from a
**larger teacher NanoUNet** will (1) reach teacher FID-R18 within **+10% relative**
and (2) beat an equal-capacity **hard** (scratch) student by ≥ **5% relative FID**.

## Why I expect this

- Cycle v2 distill (exp_092) closed the boosting gap with soft targets.
- Roadmap H-I1 / I-T1 ports that pattern to DDPM noise prediction.
- Phase H (`exp_102`) already established a viable NanoUNet + FID-R18 floor.

## What would prove me wrong

- Distill FID > teacher × 1.10
- Distill FID ≥ hard × 0.95 (no ≥5% relative win vs hard)
- Collapse / OOM on RTX 4060

## Metrics I will measure

- [x] Teacher / hard / distill param counts
- [x] FID-R18 for teacher, hard student, distill student (vs same val ref)
- [x] Distill / teacher FID ratio
- [x] Relative FID win vs hard: `1 − FID_distill / FID_hard`
- [x] Final train losses; wall-clock; device

## Success criteria

- **Primary (H-I1):** `FID_distill ≤ FID_teacher × 1.10` **and**
  `1 − FID_distill / FID_hard ≥ 0.05`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Teacher is a wider NanoUNet (same family), not SD / external UNet
- FID-R18 (ResNet18 pools), not Inception FID
- Curriculum (H-I2) and GV-ALR (H-I3) deferred to exp_105 / exp_105b
