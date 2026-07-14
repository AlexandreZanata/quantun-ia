# Hypothesis — EXP 109: Circuit-cut latent 6q (Phase J / H-Q3.4)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`) + PennyLane CPU fragments  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `cifar10_v1`, an **effective 6-qubit** residual guidance head on frozen **NanoVAE**
latents — implemented as **two overlapping 4q fragments** (circuit-cut, exp_091 pattern)
— will reach **FID-R18 within +1.0** of an equal-budget classical latent MLP
(parity). Absolute FID may remain high; relative latent-head parity is the claim.

## Why I expect this

- Roadmap H-Q3.4 / J-T6: reuse circuit-cut infrastructure (H-Q2.5 ✅) in latent space.
- H-Q3.1 (4q residual) already showed relative parity/advantage; 6q effective width
  via cut should match classical without barren-plateau / VRAM blow-up on 4060.
- Overlap wires [2:4] share information between fragments like exp_091.

## What would prove me wrong

- `FID_cut > FID_classical + 1.0` (parity fail)
- Collapse / OOM / PennyLane blow-up
- Both FIDs ≈ noise (VAE / latent DDPM broken)

## Metrics I will measure

- [x] NanoVAE recon MSE (train)
- [x] Classical latent-DDPM FID-R18
- [x] Circuit-cut 6q residual FID-R18
- [x] Δ FID = cut − classical (lower better for quantum)
- [x] Params; wall-clock; device

## Success criteria

- **Primary (H-Q3.4 parity):** `FID_cut ≤ FID_classical + 1.0`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Latent DDPM (not pixel NanoUNet); FID-R18 not Inception
- QNN fragments on `default.qubit` (CPU); VAE / classical path on CUDA
- Caption / T2I (H-T4 / G-T3) out of scope
