# Hypothesis — EXP 096: GoBug streaming nano (C-T6)

**Date:** 2026-07-12  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)  
**Cycle:** Research v2 · Phase C (GoBug streaming)

## What I expect to happen

On `code_defects_gobug_v1` (sha-ordered temporal train), a **ResidualNanoMLP** trained
in **chronological streaming batches** (commit-time online fine-tune) will reach val
**PR-AUC ≥ joint ResidualNano − 1.0 pp** (parity under the online constraint).

## Why I expect this

- GoBug train is already sorted by commit sha (temporal proxy); streaming mimics
  arriving defect risk batches without a full offline join.
- Continual agro year-by-year failed (exp_098); GoBug may be smoother within-repo.
- Cycle v2 Phase C still lists GoBug streaming as the unfinished benchmark arm.

## What would prove me wrong

- Streaming PR-AUC &lt; joint − 1.0 pp
- Collapse / OOM on RTX 4060
- Streaming ≪ logistic (not just joint)

## Metrics I will measure

- [x] Joint ResidualNano val PR-AUC
- [x] Streaming ResidualNano val PR-AUC
- [x] Δ pp streaming − joint
- [x] LogisticRegression val PR-AUC (honesty)
- [x] Wall-clock; trainable params

## Success criteria

- **Primary (C-T6):** streaming ≥ joint − **1.0 pp** PR-AUC
- `make check` green; ci smoke only in tests (no `tests/real/`)

## Known limitations

- Sha sort is a temporal proxy, not true commit timestamps
- Naive fine-tune (no replay/EWC)
- GoBug research benchmark — not production defect triage
