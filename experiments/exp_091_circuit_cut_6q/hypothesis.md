# Hypothesis — EXP 091: Circuit-cut 6q effective head (H-Q2.5)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase B (H-Q2.5)

## What I expect to happen

On frozen **exp_092 distill ResidualNano** bottleneck features (`acyd_maize_brazil_v1`),
an **effective 6-qubit head** implemented as **two overlapping 4-qubit fragments**
(circuit-cut style) will reach val ROC-AUC ≥ classical linear head − **1.0 pp**.

## Why I expect this

- Full 6q StronglyEntanglingLayers risks barren plateaus / VRAM on a laptop 4060.
- Circuit cutting approximates a wider circuit with two 4q TorchLayers the 4060 can run.
- Prior 4q hybrids already sit near classical parity; 6q effective width may recover
  ranking capacity without a monolithic 6q device.

## What would prove me wrong

- Circuit-cut AUC &lt; classical head − 1.0 pp
- Fragments collapse / flat loss / OOM on RTX 4060
- Cut head ≪ plain 4q by a large margin (cut approximation too lossy)

## Metrics I will measure

- [x] Classical linear head val ROC-AUC (frozen distill features)
- [x] Plain 4q re-upload head val ROC-AUC (honesty)
- [x] Circuit-cut 2×4q (effective 6q) val ROC-AUC
- [x] Δ pp cut vs classical; Δ pp cut vs plain 4q
- [x] Trainable params; wall-clock

## Success criteria

- **Primary (H-Q2.5):** cut ≥ classical − **1.0 pp**
- `make check` green; ci smoke only in tests

## Known limitations

- Frozen distill backbone; PennyLane TorchLayer on CPU
- Soft circuit-cut (overlapping fragments), not full qudit tomography reconstruction
- Hybrid fine-tune row budget capped for QNN wall-time on 4060
- Agro research benchmark — not operational planting advice
