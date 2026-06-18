# Testing

## Test Pyramid

```
tests/
├── unit/           # Pure functions, mocked dependencies
├── smoke/          # Import and forward-pass checks
├── integration/    # exp_001 CI profile smoke + golden bounds
├── contracts/      # JSONL schema validation (jsonschema)
├── real/           # RTX 4060 gate — actual training on real datasets (local only)
└── regression/     # golden_ci.json metric ranges
```

## Two-tier validation

| Tier | Command | Environment | Purpose |
|------|---------|-------------|---------|
| **1 — CI gate** | `make check` | GitHub Actions (CPU) + local lint/mypy | Wiring, schemas, golden bounds — not scientific truth |
| **2 — Real gate** | `make check-real` | **Local RTX 4060** (`QML_DEVICE=cuda`) | Train on real UCI data; mandatory before release |

Real tests use `@pytest.mark.real` and **skip** when CUDA is unavailable (CI never runs them).

```bash
make health-gpu          # verify NVIDIA GPU before long runs
make check-real          # 6 real tests: nanotrainer + exp_026 + publication bounds
make check-real VERBOSE=1
```

Publication-profile experiment runs (`make phase-c-publication`) are manual and logged to `results.md`.

## Running Tests

```bash
# Full engineering gate (lint + mypy + tests + contracts)
make check

# Real hardware gate (RTX 4060 required)
make check-real

# Local pytest
source .venv/bin/activate
MLFLOW_DISABLE=1 QML_DEVICE=cuda pytest tests/ -v

# With coverage (≥ 80%) — excludes tests/real/
MLFLOW_DISABLE=1 QML_DEVICE=cuda pytest tests/ --ignore=tests/real --cov=src --cov-fail-under=80

# Docker
make test
```

## Coverage Thresholds

Configured in `pyproject.toml`:

| Metric | Minimum |
|--------|---------|
| Statements | 80% |

## Type Checking

```bash
make typecheck    # mypy src/training src/quantum
```

## Pre-commit

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hooks: ruff, mypy (training + quantum), trailing whitespace, hypothesis placeholder check.

## Contract Tests

`tests/contracts/test_jsonl_schema.py` validates sample and live `logs/experiments.jsonl`
records against JSON schemas in `tests/contracts/jsonl_schema.py`.

## CI Pipeline

GitHub Actions jobs: **Lint**, **Type Check**, **Unit Tests** (80% cov), **Experiment Smoke**,
**Contracts**, **pip-audit**, **Editable Install**, **Docker Test Suite**, **E2E**.

## Test Files (selected)

| File | What it tests |
|------|---------------|
| `smoke/test_imports.py` | All core modules import without error |
| `unit/test_metrics.py` | ExperimentLogger writes valid JSON |
| `unit/test_config.py` | Config loader merges defaults and profiles |
| `contracts/test_jsonl_schema.py` | JSONL record schema compliance |
| `integration/test_exp_001_smoke.py` | CI profile golden bounds |
| `real/test_gpu_nanotrainer.py` | Classical MLP + hybrid on breast cancer/Pima (CUDA) |
| `real/test_exp_026_api_cli_parity.py` | Async API vs CLI holdout parity (exp_026) |
| `real/test_publication_bounds.py` | 10-seed hybrid mean within exp_024/025 publication CI (CUDA) |
| `unit/test_adaptive_lr.py` | Gradient-variance LR scaling |
| `unit/test_cli.py` | `qml-run` CLI entry point |

## Health Check

```bash
make health       # disk, logs writable, MLflow/DVC optional checks
make health-gpu   # above + NVIDIA CUDA GPU probe (RTX 4060 gate)
```
