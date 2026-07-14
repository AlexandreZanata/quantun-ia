# Hypothesis — EXP 111: Measurement-scheduled CFG (Phase J / H-Q3.6)

**Date:** 2026-07-14  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v3 · Phase J (quantum recipes 3.0)

## What I expect to happen

On `flickr8k_captions_v1`, a **measurement-scheduled guidance** sampler (timestep-annealed
Bernoulli masking of CLIP fusion channels as a classifier-free guidance substitute)
will improve OpenCLIP **CLIPScore by ≥ 0.5** absolute versus **classical CFG**
sampling on the **same** CFG-trained TinyDiT checkpoint.

## Why I expect this

- Roadmap H-Q3.6 / J-T8: CFG substitute via stochastic measurement schedule
  (≠ H-Q2.4 ECE measurement-dropout on tabular QNN).
- Mid-noise steps benefit from stronger partial “uncond” measurement masks; ends keep
  fuller conditioning — a schedule classical fixed-scale CFG lacks.
- Reuses G-T3 Flickr8k + TinyDiT / CLIP stack from exp_103/110.

## What would prove me wrong

- `CLIP_meas < CLIP_cfg + 0.5`
- Both arms ≈ noise / each other within noise
- OOM on 4060

## Metrics I will measure

- [x] CLIPScore classical CFG
- [x] CLIPScore measurement-scheduled guidance
- [x] Δ = meas − cfg
- [x] Params; wall-clock; device

## Success criteria

- **Primary (H-Q3.6):** `CLIP_meas ≥ CLIP_cfg + 0.5`
- `make check` green; tests use `profile=ci` only (no `tests/real/`)

## Known limitations

- Single shared TinyDiT + classical CLIP MLP fusion (no PennyLane at sample time)
- Measurement schedule is quantum-**inspired** channel masking, not circuit mid-run
- 32×32 nano pixels; absolute CLIP may remain low
