# Hypothesis — EXP 068a: Seasonal Angle Encoding on ACYD (H-Q8)

**Date:** 2026-06-19  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`, PennyLane on CPU)

## What I expect to happen

Encoding cyclic in-season weather phase (sin/cos harmonics from ACYD week aggregates) via
**AngleEmbedding** in a 4-qubit re-upload head on frozen **C4** will beat **AmplitudeEmbedding**
on the same four seasonal features by **≥ +0.5 pp** val ROC-AUC, and beat the classical sigmoid
head by **≥ +0.5 pp**.

## Why I expect this (or not)

- Climate series are periodic; angle encoding is the QML default for cyclic structure.
- exp_062 hybrid head-only was −0.19 pp vs classical — large gains are unlikely.
- Amplitude encoding on 4 scalars expanded to 16-dim may not capture phase as directly.

## Pre-registered gates

| Metric | Gate |
|--------|------|
| Val ROC-AUC | angle seasonal ≥ classical head + **0.5 pp** |
| Secondary | angle seasonal ≥ amplitude seasonal + **0.5 pp** |
| Protocol | Temporal val 2019–2021; same splits as exp_060/062 |
| Backbone | Frozen `artifacts/exp_060/.../best.pt` (eval classical head only) |

## Known limitations

- Cyclic features derived from scaled parquet columns (not raw DOY metadata).
- Single seed (42) publication; multi-seed deferred.
- QNN on CPU; classical head eval on CUDA.
