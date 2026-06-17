# Hypothesis — exp_019_nanotrainer_smoke

## Question

Does the Nano Trainer application path (`train_nanomodel.execute`) produce valid holdout metrics and JSONL records for every registered nanomodel on a shared real dataset?

## Expectation vs prior work

- **Different from exp_011–014:** Those experiments benchmark specific model families in isolation. This experiment validates the **productized** registry + orchestrator that powers `qml-train` and the Streamlit Nano Trainer.
- **Same training core:** Uses `train_with_holdout` and `ExperimentLogger` — results should be consistent with direct experiment scripts.

## Success criteria

- All tabular registry models train on `breast_cancer` with `ci` profile without error.
- `transformer_qnn_fusion` trains on `sequential_phase` with `ci` profile.
- Each run appends a line to `logs/experiments.jsonl` with `exp_id=nano_train`.
- Holdout accuracy ∈ [0.35, 1.0] per model (ci subsample).

## Failure criteria

- Any model fails to build or train.
- JSONL missing `nano_train` records.
- Accuracy outside sanity bounds (data leakage or broken pipeline).
