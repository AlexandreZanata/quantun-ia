# Hypothesis — EXP 030: Publication Large Scale Stability (30 Seeds)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

On **circles** (n=1000, noise=0.2), `hybrid_sandwich` (4 qubits, 2 layers, re-upload) trained with the
**same 30-seed protocol as exp_024** will produce a holdout mean accuracy that differs from the
**10-seed `publication_large` reference mean** by **≤ 2 percentage points** — matching exp_024's
`parity_threshold_pp` gate for continuous monitoring at larger sample size.

## Why I expect this

- `publication_large` (n=1000) narrows bootstrap CIs vs n=500 (`docs/publication_large_summary.md`).
- exp_024 proved the 2 pp tolerance is meaningful for hybrid monitoring on breast cancer (30 seeds).
- Adding 20 seeds to an already-stable 10-seed mean should not shift the estimate by more than 2 pp.

## What would prove me wrong

- |mean₃₀ − mean₁₀| > 2 pp on circles at n=1000
- Any single seed holdout accuracy < 45% (training collapse / barren plateau)
- 30-seed run time exceeds practical weekly monitoring budget (> 45 min on RTX 4060)

## Metrics I will measure

- [ ] Per-seed holdout accuracy (30 seeds)
- [ ] Reference mean (first 10 seeds) vs full 30-seed mean
- [ ] |Δmean| in percentage points
- [ ] Wall-clock elapsed on CUDA

## Success criteria

- |mean₃₀ − mean₁₀| ≤ 2.0 pp
- `make check-real` stays green after merge
- `nanotrainer` profile `publication_large` available for weekly challenger runs

## Known limitations

- Synthetic circles only; not a clinical claim
- Hybrid quantum block runs on CPU (PennyLane); classical head on GPU
- Reference band is internal (10 vs 30 seeds), not a cross-dataset comparison to exp_024 BC numbers
