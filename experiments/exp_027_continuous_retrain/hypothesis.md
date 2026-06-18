# Hypothesis — EXP 027: Continuous Retrain Champion/Challenger Gate

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

Over **4 simulated weekly retrain cycles**, a `hybrid_sandwich` challenger trained on Wisconsin Breast Cancer
(full UCI holdout) will stay **within 0.5 percentage points (pp)** of the current champion holdout accuracy
in at least **3 of 4** cycles, and the promotion gate will **block** promotion when holdout regresses by
**> 1.0 pp** vs the champion.

## Why I expect this

exp_024 established stable hybrid holdout (~97% mean over 30 seeds). Weekly retrain on the same fixed split
with different seeds should produce challengers in the same band unless pipeline drift or overfitting occurs.
The champion symlink + manifest gate makes rollback auditable via `artifacts/champion/manifest.json`.

## What would prove me wrong

- More than 1 cycle blocked due to > 1 pp regression (data or training instability)
- Challenger promoted when holdout regresses > 0.5 pp (gate bug)
- Champion checkpoint symlink missing or points to wrong seed after promotion
- `make check-real` fails after continuous loop (regression in training stack)

## Metrics I will measure

- [x] Per-cycle challenger holdout accuracy vs champion
- [x] Promotion / blocked / kept decisions
- [x] `artifacts/champion/manifest.json` + `checkpoint` symlink validity
- [x] Append-only lines in `logs/experiments.jsonl` for each retrain

## Success criteria

- 4 weekly cycles complete on RTX 4060 (`profile=ci` for smoke, `publication` for local gate)
- Promotion gate matches unit tests (`should_promote`, `should_rollback`)
- `results.md` documents per-cycle table and verdict

## Known limitations

- Simulated weeks (sequential runs), not actual cron — cron wiring lives in `.local/scripts/`
- Single dataset (breast cancer); Pima generalization deferred to exp_031
- Not a clinical deployment claim — research prototype only
