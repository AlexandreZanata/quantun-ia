# Hypothesis — EXP 108: Quantum flow coupling in TinyDiT (Phase J / H-Q3.3)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `cifar10_v1`, inserting a **unitary (quantum-inspired) flow coupling** in the
TinyDiT mid-block will improve **FID-R18 by ≥ 5% relative** versus an identical
TinyDiT whose mid-block uses a **classical affine coupling** (same capacity budget).

## Why I expect this

- Roadmap H-Q3.3 / J-T5: new mechanism (unitary mixing), not H-Q2.x rehashes.
- Volume-preserving unitary coupling can remix patch features without collapsing
  magnitude the way unconstrained affine scale stacks may.
- Class-cond TinyDiT I2I on CIFAR avoids waiting on G-T3 captions (H-T4 T2I remains open).

## What would prove me wrong

- `FID_unitary > FID_classical × 0.95` (relative win gate fails)
- Collapse / OOM on 4060
- Both FIDs ≈ noise (TinyDiT floor broken)

## Metrics I will measure

- [x] Classical-coupling TinyDiT FID-R18
- [x] Unitary-coupling TinyDiT FID-R18
- [x] Relative FID improvement `(FID_c − FID_u) / FID_c`
- [x] Params (each arm); wall-clock; device

## Success criteria

- **Primary (H-Q3.3):** `FID_unitary ≤ FID_classical × (1 − 0.05)`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Unconditional / class-free noise prediction on CIFAR pixels (text-cond H-T4 out of scope)
- Unitary coupling is quantum-**inspired** (Givens / SO(2) blocks on GPU), not PennyLane shots
- FID-R18 not Inception-v3
