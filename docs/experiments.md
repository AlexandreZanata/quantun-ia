# Experiments

## Overview

Ten experiments compare classical and quantum ML approaches on controlled synthetic tasks.
Configuration is centralized in `config/experiments.yaml`.

| ID | Name | Question |
|----|------|----------|
| 001 | Quantum vs Classical | Which learns faster on binary classification? |
| 002 | Hybrid Architecture | Does combining classical + quantum beat both alone? |
| 003 | Entanglement Effect | Does CNOT entanglement help or hurt learning? |
| 004 | Data Poisoning | Angle vs amplitude encoding under label noise? |
| 005 | Curriculum Quantum | Does staged easy→hard training beat random order? |
| 006 | Barren Plateau | How does gradient variance scale with qubit count? |
| 007 | Self-Play Quantum | Can a QNN improve by re-training on hard examples? |
| 008 | Data Re-upload | Does re-uploading beat basic QNN at matched qubits? |
| 009 | Entanglement Basic | Does entanglement help without re-upload? |
| 010 | Poison Re-upload Ablation | Do fewer layers or lower LR improve poison robustness? |

**Publication profile defaults:** `circles`, `noise=0.2`, `n_samples=500`, **10 seeds**, 30% holdout.

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

**Models:** `classical_8`, `classical_32`, `quantum_4q_2l`, `quantum_6q_3l`, `quantum_reupload_4q_3l`  
**Eval:** 30% holdout, 10 seeds, bootstrap 95% CI  
**Result:** classical_32 best on n=500; see `results.md` for publication_large profile

### EXP 002 — Hybrid Architecture

**Architectures:** HybridSandwich, QuantumFirst, ClassicalFirst (re-upload QNN)  
**Eval:** 30% holdout, 10 seeds  
**Result:** No clear hybrid winner at n=500; QuantumFirst matches classical_32 at n=1000

### EXP 003 — Entanglement Effect

**Variants:** `none`, `chain`, `chain_half`, `ring` (re-upload QNN)  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `none`  
**Result:** `none` leads; no Holm-significant topology effect

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
**Metric:** Mean gradient variance (concatenated parameter gradients, 50 random inits)  
**Implementation:** `src/training/gradients.py`

### EXP 007 — Self-Play Quantum

**Loop:** Predict on train pool → select misclassified → fine-tune → repeat (5 rounds)  
**Eval:** holdout test accuracy each round (no train/eval leakage)  
**Applicability gate:** base model must exceed `learnability_threshold` (0.55)

### EXP 008 — Data Re-upload

**Models:** `quantum_basic`, `quantum_reupload`, `classical_matched` (parameter-matched)  
**Eval:** 30% holdout, 10 seeds, paired Wilcoxon vs basic and classical  
**Implementation:** `src/quantum/qnn_reupload.py`, `src/training/param_match.py`

### EXP 009 — Entanglement Basic

**Variants:** `none`, `chain`, `chain_half`, `ring` (basic QNN, no re-upload)  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `none`  
**Goal:** Isolate entanglement effect without re-upload expressivity

### EXP 010 — Poison Re-upload Ablation

**Variants:** `reupload_3l`, `reupload_2l`, `reupload_lr_low`  
**Poison rates:** 0%, 30%  
**Eval:** 30% holdout, 10 seeds, Holm-corrected Wilcoxon vs `reupload_3l` baseline  
**Goal:** Test whether depth or learning rate fixes re-upload under poisoning

## Known Flags & Mitigations

| Flag | Mitigation applied |
|------|-------------------|
| exp_006 NaN gradients | Fixed: concatenate all param gradients before `.var()` |
| curriculum_margin ~50% | Replaced with `margin_batches` staged training |
| quantum poison collapse | Added amplitude encoding comparison + clean holdout eval |
| self-play 96% suspicious | Eval moved to held-out test set (30% split) |
| exp_005 curriculum worse at n=1000 | Documented as negative result in `results.md` |

## Suggested Ablations

| Experiment | Ablation |
|------------|----------|
| EXP 003 | `chain_half` vs full `chain` (already in config) |
| EXP 004 | `angle` vs `amplitude` under poison (already in config) |
| EXP 005 | Increase `curriculum_stages` or `epochs_per_stage` |
| EXP 006 | Add parameter-shift rule vs autograd for deep circuits |
| EXP 007 | Reduce `rounds` to check overfitting on holdout |
| EXP 008 | Vary `n_layers` for re-upload (see exp_010) |
| EXP 009 | Compare with exp_003 (re-upload vs basic entanglement) |
| EXP 010 | Add amplitude encoding baseline under same poison rates |

## Publication Profiles

| Profile | `n_samples` | Seeds | Command |
|---------|-------------|-------|---------|
| `publication` | 500 | 10 | `python experiments/exp_NNN/run.py` |
| `publication_large` | 1000 | 10 | `QML_PROFILE=publication_large python ...` or `make experiment-large` |

## References

- [PennyLane Docs](https://docs.pennylane.ai/)
- [QML Book (Schuld)](https://arxiv.org/abs/2103.09522)
- [Barren Plateaus Demo](https://pennylane.ai/qml/demos/tutorial_barren_plateaus/)
