# Quantum-Inspired Micro ML Lab

A complete laboratory for learning ML with quantum techniques, rigorous measurement, and Cursor as a training co-pilot.

## Quick Start

```bash
# With Docker (recommended)
make docker-build
make test
make experiment
make dashboard

# Local setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests/ -v
python experiments/exp_001_quantum_vs_classical/run.py
streamlit run dashboard/app.py
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/getting-started.md) | Setup, Docker, first experiment |
| [Architecture](docs/architecture.md) | Project structure and design |
| [Experiments](docs/experiments.md) | All 7 experiments explained |
| [Hypothesis Workflow](docs/hypothesis-workflow.md) | How to run experiments rigorously |
| [Testing](docs/testing.md) | Test strategy and CI |
| [Docker](docs/docker.md) | Container usage reference |

## Project Structure

```
quantun-ia/
├── src/           # Models, data, training utilities
├── experiments/   # exp_001 – exp_007 + template
├── tests/         # Unit + smoke tests
├── dashboard/     # Streamlit progress dashboard
├── docs/          # Full documentation
├── logs/          # experiments.jsonl (append-only)
└── config/        # experiments.yaml
```

## CI

GitHub Actions runs on every push/PR: lint (ruff), unit tests (pytest + coverage ≥ 70%), Docker test suite, smoke imports.

## License

MIT
