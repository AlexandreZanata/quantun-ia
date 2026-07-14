# Hypothesis — EXP 106: Latent residual QNN on frozen NanoVAE (Phase J / H-Q3.1)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`) + PennyLane CPU QNN  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `cifar10_v1`, a **4-qubit re-upload residual QNN** denoising frozen **NanoVAE**
latents will reach **FID-R18 within +1.0** of an equal-budget classical latent MLP
head (parity). Advantage is claimed only if FID improves by ≥ **2.0** absolute.

## Why I expect this

- Roadmap H-Q3.1 / J-T1: new latent mechanism (not H-Q2.x pixel-head failures).
- Phase H NanoUNet floor exists; Phase I transfer arms failed — quantum must beat
  a **latent classical** row in the same table.
- Residual skip around QNN (exp_086 pattern) on angle-encoded latents is 4060-feasible.

## What would prove me wrong

- `FID_quantum > FID_classical + 1.0` (parity fail)
- Collapse / OOM / PennyLane blow-up
- VAE latent not informative (both FIDs ≈ noise)

## Metrics I will measure

- [x] NanoVAE recon MSE (train)
- [x] Classical latent-DDPM FID-R18
- [x] Quantum residual latent-DDPM FID-R18
- [x] Δ FID = quantum − classical (lower better for quantum)
- [x] Params (VAE frozen + head); wall-clock; device

## Success criteria

- **Primary (H-Q3.1 parity):** `FID_q ≤ FID_classical + 1.0`
- **Advantage (optional):** `FID_q ≤ FID_classical − 2.0`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Latent DDPM (not full NanoUNet pixel sampler); FID-R18 not Inception
- QNN on `default.qubit` (CPU); VAE / classical path on CUDA
- Caption / TinyDiT T2I (H-T4 / G-T3) out of scope
