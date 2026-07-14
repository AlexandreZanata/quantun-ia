# Hypothesis — EXP 110: Text–quantum token fusion (Phase J / H-Q3.5)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`) + PennyLane CPU QNN  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `flickr8k_captions_v1`, encoding **frozen OpenCLIP text features** through a
**4-qubit angle-embedding fusion** (then cross-attn into TinyDiT) will improve
CLIPScore by ≥ **0.5** absolute versus an identical TinyDiT whose text path is a
**classical MLP fusion** (null-quantum ablate) under equal train budget.

## Why I expect this

- Roadmap H-Q3.5 / J-T7: CLIP tokens → qubits → DiT after G-T3 captions + H-T4 floor.
- exp_103 hash embedder failed the null+0.5 gate — real CLIP features give a fairer
  classical row in the same table.
- Residual/circuit-cut wins (H-Q3.1/H-Q3.4) suggest mild quantum remix can help when
  features are already informative.

## What would prove me wrong

- `CLIP_quantum < CLIP_classical + 0.5`
- Collapse / OOM / PennyLane blow-up
- Both arms ≈ noise (TinyDiT floor broken)

## Metrics I will measure

- [x] CLIPScore classical (null-quantum) fusion
- [x] CLIPScore quantum fusion
- [x] Δ = quantum − classical (primary)
- [x] Params; wall-clock; device

## Success criteria

- **Primary (H-Q3.5):** `CLIP_q ≥ CLIP_classical + 0.5`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Frozen OpenCLIP ViT-B/32 text tower; 32×32 pixels; hash-free CLIP features only
- QNN on `default.qubit` (CPU); DiT on CUDA
- Absolute CLIP may remain low vs SOTA T2I — relative gate only
