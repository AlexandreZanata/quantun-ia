# Testing

## Test Pyramid

```
tests/
├── smoke/         # Import checks — fast, run on every CI push
├── unit/          # Pure functions, mocked dependencies
└── (integration)  # Full experiment smoke runs (future)
```

## Running Tests

```bash
# Docker (recommended)
make test

# Local
source .venv/bin/activate
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70

# Single file
pytest tests/unit/test_metrics.py -v
```

## Coverage Thresholds

Configured in `pyproject.toml`:

| Metric | Minimum |
|--------|---------|
| Statements | 70% |
| Branches | — |
| Functions | — |
| Lines | 70% |

## Test Files

| File | What it tests |
|------|---------------|
| `smoke/test_imports.py` | All core modules import without error |
| `unit/test_metrics.py` | ExperimentLogger writes valid JSON |
| `unit/test_config.py` | Config loader merges defaults |
| `unit/test_base_model.py` | TrainableMixin contract (train/predict/evaluate) |
| `unit/test_structured_log.py` | JSON structured logging |
| `smoke/test_models.py` | Forward passes for all model families |
| `unit/test_poisoning.py` | Label flip + robustness measurement |
| `unit/test_generators.py` | Dataset shape, dtype, reproducibility |
| `unit/test_curriculum.py` | Difficulty sorting, batch staging |
| `unit/test_amplitude_encoding.py` | Unit norm, padding, angle scaling |

## Linting

```bash
make lint          # Docker
ruff check src/ tests/ experiments/   # Local
ruff check --fix src/                 # Auto-fix
```

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on every push/PR:

| Job | What it does |
|-----|-------------|
| `lint` | ruff check on src/, tests/, experiments/ |
| `test` | pytest with coverage ≥ 70% |
| `docker-test` | Full test suite inside Docker container |
| `smoke-import` | Verify PennyLane, PyTorch, Streamlit imports |

## Writing New Tests

Follow TDD: write the failing test first, then implement.

```python
# tests/unit/test_my_module.py
def test_my_function_returns_expected_shape(sample_binary_data):
    X, y = sample_binary_data
    result = my_function(X)
    assert result.shape == X.shape
```

Use fixtures from `tests/conftest.py`:
- `sample_binary_data` — 50-sample binary classification dataset
- `temp_log_file` — isolated log file via monkeypatch

## Adding Tests for New Experiments

When adding a new experiment, create at minimum:
1. A smoke test that the `run.py` imports without error
2. A unit test for any new utility function in `src/`
