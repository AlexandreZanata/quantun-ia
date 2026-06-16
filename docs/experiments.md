# Experiments

## Overview

Seven experiments compare classical and quantum ML approaches on controlled synthetic tasks. Configuration is centralized in `config/experiments.yaml`.

| ID | Name | Question |
|----|------|----------|
| 001 | Quantum vs Classical | Which learns faster on binary classification? |
| 002 | Hybrid Architecture | Does combining classical + quantum beat both alone? |
| 003 | Entanglement Effect | Does CNOT entanglement help or hurt learning? |
| 004 | Data Poisoning | Angle vs amplitude encoding under label noise? |
| 005 | Curriculum Quantum | Does staged easy→hard training beat random order? |
| 006 | Barren Plateau | How does gradient variance scale with qubit count? |
| 007 | Self-Play Quantum | Can a QNN improve by re-training on hard examples? |

## Running an Experiment

```bash
# 1. Write hypothesis (required)
vim experiments/exp_003_entanglement_effect/hypothesis.md

# 2. Run
source .venv/bin/activate
python experiments/exp_003_entanglement_effect/run.py

# 3. View in dashboard
make dashboard-local    # http://localhost:8501 → [ REFRESH DATA ]

# 4. Document results
vim experiments/exp_003_entanglement_effect/results.md
```

## Experiment Details

### EXP 001 — Quantum vs Classical

**Models:** `classical_8`, `classical_32`, `quantum_4q_2l`, `quantum_6q_3l`  
**Eval:** 30% holdout, 3 seeds, mean ± std logged  
**Result:** classical_32 best (84.8% ± 1.9% holdout)

### EXP 002 — Hybrid Architecture

**Architectures:** HybridSandwich, QuantumFirst, ClassicalFirst  
**Eval:** 30% holdout, 3 seeds  
**Result:** All ~82–83% holdout, no clear hybrid winner

### EXP 003 — Entanglement Effect

**Variants:** `none`, `chain`, `chain_half`, `ring`  
**Eval:** 30% holdout, 3 seeds  
**Result:** ring best (80.0%), chain_half worst (66.3% ± 10.9%)

### EXP 004 — Data Poisoning

**Poison rates:** 0%, 5%, 10%, 20%, 30%  
**Encodings:** `angle` (QuantumNetBasic) vs `amplitude` (QuantumNetAmplitude)  
**Eval:** train on poisoned labels, evaluate on **clean holdout test set** (30% split)  
**Compares:** Classical MLP vs both quantum encodings

### EXP 005 — Curriculum Quantum

**Methods:**
- `random` — shuffled baseline (eval on holdout)
- `margin_batches` — staged easy→hard batches (`curriculum_stages: 4`, `epochs_per_stage: 12`)

> Global margin ordering alone caused poor results (~50% acc). Batched curriculum fixes exposure bias.

### EXP 006 — Barren Plateau

**Qubit counts:** 2, 4, 6, 8, 10  
**Metric:** Mean gradient variance (concatenated parameter gradients, 20 random inits)  
**Implementation:** `src/training/gradients.py`

### EXP 007 — Self-Play Quantum

**Loop:** Predict on train pool → select misclassified → fine-tune → repeat (5 rounds)  
**Eval:** holdout test accuracy each round (no train/eval leakage)

## Known Flags & Mitigations

| Flag | Mitigation applied |
|------|-------------------|
| exp_006 NaN gradients | Fixed: concatenate all param gradients before `.var()` |
| curriculum_margin ~50% | Replaced with `margin_batches` staged training |
| quantum poison collapse | Added amplitude encoding comparison + clean holdout eval |
| self-play 96% suspicious | Eval moved to held-out test set (30% split) |

## Suggested Ablations

| Experiment | Ablation |
|------------|----------|
| EXP 003 | `chain_half` vs full `chain` (already in config) |
| EXP 004 | `angle` vs `amplitude` under poison (already in config) |
| EXP 005 | Increase `curriculum_stages` or `epochs_per_stage` |
| EXP 006 | Add parameter-shift rule vs autograd for deep circuits |
| EXP 007 | Reduce `rounds` to check overfitting on holdout |

## 4-Week Roadmap

| Week | Experiments | Goal |
|------|-------------|------|
| 1 | EXP 001 | Setup + understand QNN vs classical speed |
| 2 | EXP 002, 003 | Architecture and entanglement |
| 3 | EXP 004, 005 | Robustness and curriculum |
| 4 | EXP 006, 007 | Barren plateau + self-play |

## References

- [PennyLane Docs](https://docs.pennylane.ai/)
- [QML Book (Schuld)](https://arxiv.org/abs/2103.09522)
- [Barren Plateaus Demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/)
