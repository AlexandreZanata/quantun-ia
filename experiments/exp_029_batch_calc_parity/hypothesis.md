# Hypothesis — EXP 029: Batch Calculation vs API Parity

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

`scripts/batch_predict.py` scoring the **full Wisconsin Breast Cancer dataset (569 rows, raw features)**
will produce probabilities **within 1e-5** of `POST /api/v1/predictions` row-for-row on the same
checkpoint bundle (`quantum_nano_bc_app`, seed 42).

## Why I expect this

Batch and API both route through `predict_nanomodel.execute` with identical DTOs; exp_028 validated
the chatbot adapter surface. Chunked batch inference only concatenates chunk results.

## What would prove me wrong

- Any row with max \|Δp\| ≥ 1e-5 between batch CSV output and API
- Wrong feature count accepted (≠ 30 columns)
- Double-scaling (raw features scaled twice)
- Output header missing `exp_id`, `seed`, or checkpoint metadata

## Metrics I will measure

- [x] max \|Δp\| across all 569 rows (batch vs API)
- [x] Row count parity (569 in / 569 out)
- [x] Output header reproducibility fields
- [x] Wall-clock batch latency on RTX 4060

## Success criteria

- 569/569 rows within 1e-5 probability tolerance
- `make check-real` remains green (10/10 with new gate test)

## Known limitations

- Single dataset (breast cancer); not a clinical deployment claim
- CSV input only in local workflow; JSON supported for automation
