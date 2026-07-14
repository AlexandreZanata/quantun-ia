# Hypothesis — EXP 107: Patch amplitude bottleneck (Phase J / H-Q3.2)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`) + PennyLane CPU QNN  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `cifar10_v1`, an **amplitude-encoded 4×4 RGB-patch QNN bottleneck** (4 qubits,
unit-norm amp features) will reconstruct images with **FID-R18 within ±1.0** of an
equal-width **classical linear bottleneck** (parity).

## Why I expect this

- H-Q3.1 latent residual parity unlocked J-T3 → next H-Q3.2.
- 4×4 patches flatten to 48 dims → project to 16 (= 2^4) for AmplitudeEmbedding
  with mandatory unit-norm (encoding contract).
- Classical `Linear(48→16→48)` matched bottleneck width keeps the comparison fair.

## What would prove me wrong

- `|FID_q − FID_classical| > 1.0`
- Collapse / OOM / amplitude norm failures
- Dead QNN (FID ≫ classical and ≫ noise baseline)

## Metrics I will measure

- [x] Train patch MSE (classical / quantum)
- [x] FID-R18 of full-image reconstructions vs val
- [x] `|Δ FID|` vs classical
- [x] Params; wall-clock; device

## Success criteria

- **Primary (H-Q3.2):** `|FID_q − FID_classical| ≤ 1.0`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Patch AE reconstruction FID (not generative DDPM)
- QNN on `default.qubit` (CPU); classical path on CUDA
- RGB→16 projection discards some color detail before amplitude encoding
