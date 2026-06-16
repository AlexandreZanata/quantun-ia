# Experiments

## Overview

Seven experiments compare classical and quantum ML approaches on controlled synthetic tasks.

| ID | Name | Question |
|----|------|----------|
| 001 | Quantum vs Classical | Which learns faster on binary classification? |
| 002 | Hybrid Architecture | Does combining classical + quantum beat both alone? |
| 003 | Entanglement Effect | Does CNOT entanglement help or hurt learning? |
| 004 | Data Poisoning | Are QNNs more or less robust to label noise? |
| 005 | Curriculum Quantum | Does easy-to-hard ordering improve QNN training? |
| 006 | Barren Plateau | How does gradient variance scale with qubit count? |
| 007 | Self-Play Quantum | Can a QNN improve by re-training on its own hard examples? |

## Running an Experiment

```bash
# 1. Write hypothesis
vim experiments/exp_003_entanglement_effect/hypothesis.md

# 2. Run
python experiments/exp_003_entanglement_effect/run.py

# 3. View in dashboard
streamlit run dashboard/app.py

# 4. Document results
vim experiments/exp_003_entanglement_effect/results.md
```

## Experiment Details

### EXP 001 — Quantum vs Classical

**Models:** `classical_8`, `classical_32`, `quantum_4q_2l`, `quantum_6q_3l`  
**Dataset:** make_moons (200 samples, noise=0.1)  
**Metrics:** accuracy per epoch, wall-clock time, parameter count

### EXP 002 — Hybrid Architecture

**Architectures:**
- `HybridSandwich` — Classical → Quantum → Classical
- `QuantumFirst` — Quantum → Classical
- `ClassicalFirst` — Classical → Quantum

### EXP 003 — Entanglement Effect

**Variants:** `none`, `chain`, `ring` CNOT patterns  
**Key question:** Does more entanglement improve representational capacity or worsen gradient flow?

### EXP 004 — Data Poisoning

**Poison rates:** 0%, 5%, 10%, 20%, 30%  
**Compares:** Classical MLP vs QuantumNetBasic degradation curves

### EXP 005 — Curriculum Quantum

**Methods:** `margin` (easy-first by centroid distance) vs `random` baseline

### EXP 006 — Barren Plateau

**Qubit counts:** 2, 4, 6, 8, 10  
**Metric:** Mean gradient variance across random initializations

### EXP 007 — Self-Play Quantum

**Loop:** Predict → select misclassified → fine-tune → repeat (5 rounds)

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
