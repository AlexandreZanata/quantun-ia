# Testing

## Test Pyramid

```
tests/
├── unit/           # Pure functions, mocked dependencies
├── smoke/          # Import and forward-pass checks
├── integration/    # exp_001 CI profile smoke + golden bounds
├── contracts/      # JSONL schema validation (jsonschema)
└── regression/     # golden_ci.json metric ranges
```

## Running Tests

```bash
# Full engineering gate (lint + mypy + tests + contracts)
make check

# Local pytest
source .venv/bin/activate
MLFLOW_DISABLE=1 pytest tests/ -v

# With coverage (≥ 80%)
MLFLOW_DISABLE=1 pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

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
**Contracts**, **pip-audit**, **Editable Install**, **Docker Test Suite**.

## Test Files (selected)

| File | What it tests |
|------|---------------|
| `smoke/test_imports.py` | All core modules import without error |
| `unit/test_metrics.py` | ExperimentLogger writes valid JSON |
| `unit/test_config.py` | Config loader merges defaults and profiles |
| `contracts/test_jsonl_schema.py` | JSONL record schema compliance |
| `integration/test_exp_001_smoke.py` | CI profile golden bounds |
| `unit/test_adaptive_lr.py` | Gradient-variance LR scaling |
| `unit/test_cli.py` | `qml-run` CLI entry point |

## Health Check

```bash
make health    # disk, logs writable, MLflow/DVC optional checks
```
