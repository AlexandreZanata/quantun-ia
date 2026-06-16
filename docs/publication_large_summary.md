# Publication Large Profile — Summary (n=1000)

**Run date:** 2026-06-16  
**Profile:** circles, noise=0.2, n=1000, 10 seeds  
**Command:** `make experiment-large` (exp_006 re-run after gradients fix)

## Holdout leaderboard (mean %, 95% CI)

| Exp | Best model | Mean | CI |
|-----|------------|------|-----|
| 001 | classical_32 | **67.6%** | [65.7%, 69.6%] |
| 002 | quantum_first | **67.6%** | [65.9%, 69.4%] |
| 003 | entanglement_none | **66.3%** | [63.0%, 69.0%] |
| 004 | classical_poison_0 | **66.0%** | [63.4%, 68.7%] |
| 005 | curriculum_random | **63.0%** | [59.7%, 66.3%] |
| 007 | self_play_base | **62.9%** | [59.3%, 66.1%] |
| 008 | classical_matched_h9 | **62.8%** | [60.5%, 65.2%] |
| 009 | entanglement_chain_half | **65.1%** | [62.3%, 67.8%] |
| 010 | reupload_3l_poison_0 | **60.8%** | [58.1%, 63.9%] |

## Key deltas vs n=500

- CIs ~20–30% narrower across all experiments.
- **exp_002** QuantumFirst matches classical_32 from exp_001 (67.6%).
- **exp_005** curriculum margin_batches **worse** than random (54.9% vs 63.0%) at n=1000.
- **exp_007** self-play: best = base (62.9%), zero gain.
- **exp_009** basic QNN: chain_half leads (65.1%), still no Holm-significant topology effect.

## exp_006 (gradient diagnostics, profile-independent)

| Qubits | Grad variance | 95% CI |
|--------|---------------|--------|
| 2 | 0.0356 | [0.030, 0.041] |
| 4 | 0.0151 | [0.012, 0.018] |
| 6 | 0.0129 | [0.011, 0.015] |
| 8 | 0.0080 | [0.006, 0.010] |
| 10 | 0.0063 | [0.005, 0.007] |

Barren plateau confirmed (variance decays with qubit count). Fixed `batch_size=1` BCE shape bug for parameter-shift path.
