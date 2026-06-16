# Results — EXP 001

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 50 epochs

## What happened

| Model | Mean holdout acc | Std |
|-------|------------------|-----|
| classical_32 | **84.8%** | ±1.9% |
| classical_8 / quantum_6q_3l | 82.6% | ±1.9–2.6% |
| quantum_4q_2l | 79.3% | ±3.7% |

Classical MLP (32 hidden) achieved the best generalization. Quantum models matched or trailed classical, with higher variance across seeds. Quantum training was ~20× slower per epoch.

## Comparison with hypothesis

If the hypothesis expected quantum to learn faster or generalize better, it was **not supported** on this dataset with these circuit depths. Classical models are simpler and sufficient for 2D moons.

## Unexpected finding

`quantum_6q_3l` tied `classical_8` on holdout (82.6%) despite more parameters — more qubits did not mean better generalization here.

## Suggested next experiment

- Try angle encoding with data re-uploading, or a harder dataset (circles, higher noise)
- Compare with `exp_004` poisoning robustness to see if quantum offers any advantage under noise
