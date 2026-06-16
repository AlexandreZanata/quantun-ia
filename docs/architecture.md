# Architecture

## Overview

The project follows a layered structure separating data, models, training, and experiments.

```
src/
├── classical/     # Classical ML models (Perceptron, TransformerMini, RNNMini)
├── quantum/       # Quantum models (QNN, HybridModel, AmplitudeEncoding)
├── data/          # Dataset generators, augmentation, poisoning
└── training/      # Universal trainer, metrics logger, curriculum learning

experiments/
└── exp_NNN_<name>/
    ├── hypothesis.md   # Written BEFORE running
    ├── run.py          # Experiment entry point
    └── results.md      # Filled AFTER running

logs/
└── experiments.jsonl   # Append-only central log

dashboard/
└── app.py              # Streamlit visualization
```

## Module Responsibilities

### `src/data/`

| Module | Purpose |
|--------|---------|
| `generators.py` | Synthetic binary classification datasets (moons, circles) |
| `splits.py` | Stratified train/test split (before preprocessing) |
| `augmentation.py` | Gaussian noise, label flip augmentation |
| `poisoning.py` | Intentional label corruption for robustness tests |

### `src/classical/`

| Module | Purpose |
|--------|---------|
| `perceptron.py` | Single-layer perceptron baseline |
| `mlp.py` | Multi-layer perceptron (ClassicalNet) |
| `transformer_mini.py` | Minimal transformer for sequence classification |
| `rnn_mini.py` | GRU-based binary classifier |

### `src/quantum/`

| Module | Purpose |
|--------|---------|
| `qnn_basic.py` | Basic variational quantum circuit (VQC) |
| `qnn_entangled.py` | VQC with configurable entanglement (none/chain/chain_half/ring) |
| `qnn_amplitude.py` | VQC with amplitude encoding (unit-norm required) |
| `hybrid_model.py` | Classical-Quantum hybrid architectures |
| `amplitude_encoding.py` | Data encoding utilities for quantum circuits |

### `src/training/`

| Module | Purpose |
|--------|---------|
| `trainer.py` | Universal training loop with Adam + BCELoss |
| `base_model.py` | `TrainableMixin` — enforces train/predict/evaluate contract |
| `config.py` | Loads `config/experiments.yaml` |
| `structured_log.py` | JSON structured logging via loguru |
| `metrics.py` | `ExperimentLogger` — writes to `logs/experiments.jsonl` |
| `curriculum.py` | Difficulty-based ordering and staged batch training |
| `gradients.py` | Gradient variance measurement (barren plateau) |

## Model Interface Contract

Every model inherits `TrainableMixin` and implements:

```python
def train(self, X, y, exp_id, model_name, epochs=50, lr=0.01): ...
def predict(self, X): ...
def evaluate(self, X, y) -> dict: ...
```

## Data Flow

```
generators.py → run.py → trainer.py → metrics.py → experiments.jsonl → dashboard/app.py
```

## Configuration

`config/experiments.yaml` centralizes default hyperparameters and experiment metadata. All `run.py` scripts load settings via `src/training/config.py`.

## Design Principles

1. **Hypothesis-first** — no experiment runs without a written hypothesis
2. **Append-only logs** — experiment history is never deleted
3. **Comparable metrics** — all experiments use the same `ExperimentLogger` format
4. **English-only** — all identifiers, comments, and docs in English
5. **Testable units** — data utilities and metrics are pure functions, easy to unit test
