# Architecture

## Overview

The project follows a layered structure separating data, models, training, and experiments.

```
src/
├── classical/     # Classical ML models (Perceptron, MLP, TransformerMini, RNNMini)
├── quantum/       # Quantum models (QNN variants, HybridModel, encoding utilities)
├── data/          # Dataset generators, real loaders, registry, scaling, augmentation, poisoning, splits
└── training/      # Trainer, metrics, statistics, protocol, curriculum, self-play

experiments/
└── exp_NNN_<name>/
    ├── hypothesis.md   # Written BEFORE running
    ├── run.py          # Experiment entry point (thin orchestrator)
    └── results.md      # Filled AFTER running

config/
└── experiments.yaml    # Central hyperparameters and profiles

logs/
└── experiments.jsonl   # Append-only central log

dashboard/
├── app.py              # Streamlit visualization
├── benchmark_data.py   # JSONL loader and leaderboard normalization
└── terminal_report.py  # ASCII terminal leaderboard
```

## Module Responsibilities

### `src/data/`

| Module | Purpose |
|--------|---------|
| `generators.py` | Synthetic binary classification datasets (moons, circles) |
| `real_datasets.py` | UCI tabular (breast cancer, wine, iris) and MNIST binary loaders |
| `dataset_registry.py` | Unified `get_dataset()` / `prepare_dataset()` API |
| `scaling.py` | Leakage-safe StandardScaler and PCA (fit on train only) |
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
| `qnn_basic.py` | Basic variational quantum circuit (angle encoding) |
| `qnn_reupload.py` | Data re-uploading VQC (layer-wise feature re-embedding) |
| `qnn_entangled.py` | VQC with configurable entanglement (none/chain/chain_half/ring) |
| `qnn_amplitude.py` | VQC with amplitude encoding (unit-norm required) |
| `qnn_factory.py` | Config-driven QNN builder (`basic`, `reupload`, `entangled`) |
| `hybrid_model.py` | Classical-Quantum hybrid architectures |
| `amplitude_encoding.py` | Data encoding utilities (normalize, pad, angle encode) |
| `circuit_utils.py` | Backprop vs parameter-shift gradient policy |

### `src/training/`

| Module | Purpose |
|--------|---------|
| `trainer.py` | Universal training loop with Adam + BCELoss |
| `base_model.py` | `TrainableMixin` — enforces train/predict/evaluate contract |
| `config.py` | Loads `config/experiments.yaml` with profile merging |
| `structured_log.py` | JSON structured logging via loguru |
| `metrics.py` | `ExperimentLogger` — writes to `logs/experiments.jsonl` |
| `holdout.py` | Multi-seed holdout runner and summary aggregation |
| `statistics.py` | Bootstrap CI, paired Wilcoxon, Holm-Bonferroni correction |
| `protocol.py` | Experiment protocol logging and applicability gates |
| `param_match.py` | Parameter-matched classical baselines for fair comparison |
| `curriculum.py` | Difficulty-based ordering and staged batch training |
| `self_play.py` | Capped hard-example self-play fine-tuning loop |
| `gradients.py` | Gradient variance measurement (barren plateau diagnostics) |
| `reproducibility.py` | Global seed control for numpy, torch, sklearn |
| `tracking.py` | MLflow dual-write alongside JSONL |
| `checkpoints.py` | Model artifact persistence under `artifacts/` |
| `device.py` | Auto-detect CPU/CUDA for PyTorch training |
| `hpo.py` | Optuna hyperparameter optimization wrapper |
| `plot_style.py` | Matplotlib/seaborn publication figure style |
| `adaptive_lr.py` | Gradient-variance adaptive learning rate (exp_015) — see [method_adaptive_lr.md](../docs/method_adaptive_lr.md) |
| `ci_smoke.py` | Fast CI profile runner for regression bounds |

## Model Interface Contract

Every model inherits `TrainableMixin` and implements:

```python
def train(self, X, y, exp_id, model_name, epochs=50, lr=0.01): ...
def predict(self, X): ...
def evaluate(self, X, y) -> dict: ...
```

## Data Flow — Single Experiment

```
config/experiments.yaml
        │
        ▼
dataset_registry.py / generators.py ──► splits.py ──► scaling.py (train-fit only)
        │
        ▼
   run.py (orchestrator)
        │
        ▼
   trainer.py ──► model.train() ──► model.evaluate(holdout)
        │
        ▼
   metrics.py (ExperimentLogger) ──► logs/experiments.jsonl
        │
        ▼
   dashboard/app.py
```

## Data Flow — Multi-Seed Holdout Pipeline

```
config/experiments.yaml  (seeds: [42, 123, ..., 5000])
        │
        ▼
   holdout.py
        │
        ├── seed=42  ──► split ──► train ──► holdout eval ──► log per-seed
        ├── seed=123 ──► split ──► train ──► holdout eval ──► log per-seed
        └── ...      (10 seeds)
        │
        ▼
   statistics.py
        ├── bootstrap_ci()        → mean ± 95% CI per model
        ├── paired_wilcoxon()     → effect vs baseline per seed
        └── holm_bonferroni()     → multiple-comparison correction
        │
        ▼
   metrics.py  →  multi_seed_summary / paired_comparison records in JSONL
        │
        ▼
   dashboard/benchmark_data.py  →  leaderboard table + charts
```

## Configuration

`config/experiments.yaml` centralizes default hyperparameters, experiment metadata, and profiles:

| Profile | `n_samples` | Seeds |
|---------|-------------|-------|
| `publication` | 500 | 10 |
| `publication_large` | 1000 | 10 |
| `ci` | 200 | 1 |

All `run.py` scripts load settings via `src/training/config.py`:

```python
from src.training.config import load_experiment_config
cfg = load_experiment_config("exp_008_data_reupload", profile="publication")
```

## Design Principles

1. **Hypothesis-first** — no experiment runs without a written hypothesis
2. **Append-only logs** — experiment history is never deleted
3. **Comparable metrics** — all experiments use the same `ExperimentLogger` format
4. **No data leakage** — stratified split before poisoning, curriculum, or scaling
5. **Fair baselines** — parameter-matched classical models where applicable
6. **Statistical rigor** — multi-seed runs with corrected multiple comparisons
7. **English-only** — all identifiers, comments, and docs in English
8. **Testable units** — data utilities and metrics are pure functions, easy to unit test
