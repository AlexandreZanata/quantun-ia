# Nano Trainer

The Nano Trainer is a lightweight application surface for running **mini training sessions** on **real datasets** without editing experiment folders manually.

## Surfaces

| Surface | Command |
|---------|---------|
| CLI | `qml-train --model perceptron --dataset breast_cancer --profile mini` |
| Streamlit | `make dashboard-local` → open **Nano Trainer** page |
| REST API | `make api` → `POST /api/v1/training-jobs` |
| Makefile demo | `make train-demo` |

All runs log to `logs/experiments.jsonl` with `exp_id=nano_train`. These records are **excluded** from the publication leaderboard.

## Profiles

| Profile | Use case |
|---------|----------|
| `mini` | Default for app/CLI — ~100 samples, 8 epochs, single seed |
| `ci` | Fast CI smoke tests |
| `publication` | Full holdout protocol (slower) |

Configuration lives in `config/nanotrainer.yaml` and `config/experiments.yaml`.

## Supported model × dataset pairs

Driven by `config/nanotrainer.yaml` — tabular models require tabular datasets; sequence models require sequence datasets.

**Tabular:** `perceptron`, `classical_mlp`, `quantum_angle`, `quantum_amplitude`, `hybrid_sandwich` on UCI/MNIST binary sets.

**Sequence:** `transformer_mini`, `transformer_qnn_fusion` on `sequential_binary`, `sequential_phase`.

## Examples

```bash
# Default mini profile
qml-train --model perceptron --dataset breast_cancer

# CI smoke with JSON output
qml-train --model quantum_angle --dataset breast_cancer --profile ci --json

# Sequence fusion model
qml-train --model transformer_qnn_fusion --dataset sequential_phase --epochs 10
```

## Architecture

```
qml-train / Streamlit / REST API
    → src/application/train_nanomodel.py (or create_training_job for API)
        → model_registry.build_model()
        → dataset_registry.prepare_dataset()
        → training.holdout.train_with_holdout()
    → logs/experiments.jsonl
```

## Limitations

- API training is synchronous — use `mini`/`ci` profiles for HTTP requests.
- No JWT auth yet — API uses `X-Tenant-ID` header (default `local`).
- Quantum models are CPU-bound; use `mini` or `ci` for interactive runs.
- Checkpoints are optional (`save_checkpoints=False` by default in Nano Trainer).

## Validation

- `experiments/exp_019_nanotrainer_smoke/` — batch smoke across all registry models.
- `make check` includes unit, integration, and contract tests for the Nano Trainer path.
