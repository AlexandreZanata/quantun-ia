# Hypothesis — EXP 037: Hybrid QNN Head on Frozen LargeNanoMLP (HIGGS)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

After freezing the **exp_032** `LargeNanoMLP` backbone (~1.17M params) and training only a
**4-qubit re-upload QNN head** (~300 trainable params), val ROC-AUC on HIGGS will stay within
**−1.0 pp** of the classical sigmoid head (same backbone, same splits).

## Why I expect this (or not)

- exp_002 hybrid patterns win on small clinical sets — HIGGS is harder tabular physics data.
- Replacing a tuned linear head with a tiny QNN may **not** improve AUC; barren plateau risk is low
  (4 qubits × 2 layers, shallow head only).
- Frozen backbone preserves exp_032 representation; head fine-tune is a fair ablation.

## Success criteria

- Load exp_032 checkpoint backbone; freeze trunk; train QNN head on RTX 4060 without OOM
- Report paired val AUC: **hybrid head vs classical head** (same val slice)
- **Accept:** hybrid val AUC ≥ classical − **1.0 pp** on publication profile
- **Else:** document honest negative in `results.md` (infrastructure still passes if training completes)

## Known limitations

- CI uses 8K train / 1.5K val — not full 805K train (QNN CPU bottleneck)
- PennyLane sim stays CPU; classical backbone uses CUDA
- Test split untouched; val AUC only for gate
