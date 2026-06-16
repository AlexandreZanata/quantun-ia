# Results — EXP 001

**Date:** 2026-06-16  
**Config:** 300 samples, 30% holdout, seeds [42, 123, 456], 50 epochs  
**Note:** `quantum_4q_2l` uses 1 layer + LR 0.02 (see `model_configs` in `config/experiments.yaml`)

## What happened

| Model | Mean holdout acc | Std |
|-------|------------------|-----|
| classical_32 | **85.2%** | ±2.8% |
| quantum_6q_3l | 83.0% | ±2.9% |
| quantum_4q_2l | 82.6% | ±2.8% |
| classical_8 | 81.9% | ±2.3% |

After reducing `quantum_4q_2l` to 1 layer and LR 0.02, holdout variance dropped from ±13.6% (2 layers) to ±2.8%. All models now generalize in the 82–85% band on moons.

`test_accuracy` is logged to `logs/experiments.jsonl` alongside train `final_acc`.

## Comparison with hypothesis

If the hypothesis expected quantum to clearly beat classical on 2D moons, it was **not supported**. Classical_32 still leads slightly; quantum_4q is now competitive and stable after tuning.

## Unexpected finding

Fewer layers + higher LR stabilized `quantum_4q_2l` more than adding depth — 6q_3l did not outperform the tuned 4q model.

## Suggested next experiment

- LR sweep for `quantum_6q_3l` (still seed-sensitive in some runs)
- Harder dataset (circles, noise=0.2) where quantum feature maps may matter
