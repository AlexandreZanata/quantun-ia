# Hypothesis — EXP 016 (Hybrid NAS)

**Date:** 2026-06-17  
**Author:** quantun-ia lab

## What I expect to happen

Optuna neural architecture search over hybrid classical–quantum layouts will find a configuration that **beats the fixed `hybrid_sandwich` baseline** from EXP 002 by at least **2 percentage points** mean holdout accuracy (10 seeds, circles, noise=0.2).

## Why I expect this

EXP 002 compared three hand-picked architectures with shared hyperparameters. Layer count, qubit budget, and re-upload depth interact non-linearly with the quantum block placement. A small search (architecture × qubits × layers × LR) should surface a better trade-off than any single manual preset.

## What would prove me wrong

- NAS-best mean holdout ≤ `hybrid_sandwich` fixed baseline (Holm-adjusted p ≥ 0.05).
- Best trial selects the same architecture and hyperparameters as the EXP 002 default (search adds no value).
- CI profile NAS with 3 trials already matches manual baselines within 1 pp (search budget too small to matter).

## Metrics I will measure

- [x] Mean holdout accuracy (10 seeds, bootstrap 95% CI)
- [x] Optuna best trial params (logged to `logs/hpo_results.jsonl`)
- [x] Paired Wilcoxon: `nas_best` vs each EXP 002 baseline
- [x] Cohen's d effect size on primary comparison

## Ablation plan

1. Fix architecture to `hybrid_sandwich`, search only qubits/layers/LR.
2. Disable re-upload in search space (angle-only layers).
3. Halve Optuna trials (10 vs 20) to test search-budget sensitivity.
