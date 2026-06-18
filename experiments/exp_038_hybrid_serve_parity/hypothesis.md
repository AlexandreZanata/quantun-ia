# Hypothesis — EXP 038: Hybrid QNN Serve Parity on HIGGS

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

After publishing the **exp_037** `large_nano_hybrid` checkpoint to the nanotrainer serve path
(`large_nano_hybrid × higgs_v1`), **batch_predict**, **REST API**, and **`score_higgs` chatbot tool**
return identical probabilities on a 10K HIGGS holdout slice — max |Δp| **< 1e-5** (same gate as exp_033).

## Why I expect this

- exp_033 proved serve wiring for `large_nano_mlp` on HIGGS.
- Hybrid inference uses the same `predict_nanomodel` path; PennyLane stays CPU, backbone CUDA.
- `_tensor_device()` fix in L9 ensures batch/API agree on input device for hybrid models.

## Success criteria

- `ensure_large_nano_hybrid_serve_artifact` publishes scaler + checkpoint under `artifacts/exp_037/`
- Parity gate on publication profile (10K rows) passes on RTX 4060
- `make check-real` stays green (19/19)

## Known limitations

- Slower inference than pure classical MLP (QNN head on CPU per batch row group)
- Same HIGGS infrastructure validation — not a clinical deployment claim
