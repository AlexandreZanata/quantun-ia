# Hypothesis — EXP 026: Real Application E2E (API = CLI on GPU)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

For `hybrid_sandwich` on full Wisconsin Breast Cancer, an **async REST API** training job
with `device=cuda` will produce holdout accuracy **within 0.5 percentage points (pp)** of the
**synchronous CLI** path (`train_nanomodel.execute`) for the same seed, epochs, and profile.

Both paths must call the same holdout protocol (30% stratified split, preprocessing fit on train only).

## Why I expect this

- exp_020 proved API sync jobs match the Nano Trainer for `perceptron` on CPU.
- exp_024/025 established `hybrid_sandwich` as the flagship model on real clinical tabular data.
- `process_training_job` wraps the same `train_nanomodel.execute` as the CLI; only persistence and
  device context differ.

## What would prove me wrong

- Mean |CLI − API| holdout delta **> 0.5 pp** across 5 CI seeds → API layer introduces drift.
- API job fails or returns `FAILED` while CLI succeeds for the same seed → worker or device bug.
- Async job never reaches `COMPLETED` within 120 s → queue regression.

## Metrics I will measure

- [x] Holdout accuracy per seed (CLI vs API)
- [x] Paired delta in pp per seed
- [x] Mean absolute delta across seeds
- [x] Wall-clock per path (CLI vs API async)
- [x] Verdict: `accepted` (mean |Δ| ≤ 0.5 pp) / `rejected`

## Profiles

| Profile | Seeds | Epochs | Purpose |
|---------|-------|--------|---------|
| `ci` | 5 | 12 | Local real gate + exp_026 run |
| `publication` | 10 | 50 | Pre-release confirmation (optional) |
