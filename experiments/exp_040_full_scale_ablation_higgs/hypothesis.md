# Hypothesis — EXP 040: Full-Scale Methodology Ablation on HIGGS (805K)

**Date:** 2026-06-18  
**Author:** Quantum ML Lab  
**Hardware:** NVIDIA RTX 4060 Laptop GPU (`QML_DEVICE=cuda`)

## What I expect to happen

**exp_036** on a 50K slice found **no method ≥ +0.5 pp** vs Adam baseline (best: adaptive
**+0.26 pp** mean on 10 seeds). At **full 805K train / 172.5K val**, alternative trainers may
gain enough signal to beat baseline by **≥0.5 pp** — or confirm the honest negative at scale.

## Why I expect this (or not)

- More data reduces overfitting from curriculum/champion noise.
- Adaptive LR showed the largest positive delta on the slice — may strengthen at scale.
- Curriculum failed badly on the slice (−1.61 pp) — unlikely to flip at scale.

## Methods compared (paired seeds)

Same four trainers as exp_036: `baseline`, `curriculum`, `adaptive`, `champion`.

| Profile | Train / val rows | Seeds | Baseline epochs |
|---------|------------------|-------|-----------------|
| `full_scale` | 805K / 172.5K | 3 | 12 |

## Success criteria

- All four methods × 3 seeds complete on RTX 4060 without OOM (batch 2048)
- Paired comparison logged via `compare_conditions_batch`
- **Accept:** any alternative mean val AUC ≥ baseline + **0.5 pp**
- **Else:** document honest negative at full scale in `results.md`

## Known limitations

- 3 seeds (not 10) — RTX 4060 wall-clock budget
- Val AUC only — test split untouched
- Reuses exp_036 trainer implementations
