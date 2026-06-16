# Quantum-Inspired Micro ML Lab

A laboratory for classical and quantum ML experiments with rigorous measurement, append-only logging, and a retro benchmark dashboard.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone git@github.com:AlexandreZanata/quantun-ia.git
cd quantun-ia

# 2. Local environment (recommended for development)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 3. Verify setup
pytest tests/ -v

# 4. Write hypothesis, then run an experiment
vim experiments/exp_001_quantum_vs_classical/hypothesis.md
python experiments/exp_001_quantum_vs_classical/run.py

# 5. View benchmarks (terminal + browser)
make dashboard-local
# → http://localhost:8501
```

## Docker (alternative)

```bash
make docker-build
make test
make experiment      # runs EXP 001
make dashboard       # Streamlit on :8501
```

## Run All Experiments

```bash
source .venv/bin/activate
for exp in experiments/exp_*/run.py; do python "$exp"; done
python dashboard/terminal_report.py   # ASCII leaderboard
make dashboard-local                  # interactive charts
```

Results append to `logs/experiments.jsonl` (never delete — append only).

## Documentation

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/getting-started.md) | Full setup, Makefile, workflow |
| [Experiments](docs/experiments.md) | All 7 experiments + ablations |
| [Hypothesis Workflow](docs/hypothesis-workflow.md) | Mandatory hypothesis-first discipline |
| [Architecture](docs/architecture.md) | Code structure and data flow |
| [Testing](docs/testing.md) | pytest, coverage, CI |
| [Docker](docs/docker.md) | Container reference |

## Project Structure

```
quantun-ia/
├── src/              # Models, data, training utilities
├── experiments/      # exp_001 – exp_007 + template
├── config/           # experiments.yaml (central hyperparameters)
├── dashboard/        # Retro Streamlit benchmark monitor
├── logs/             # experiments.jsonl (append-only)
├── tests/            # Unit + smoke tests
└── docs/             # Full documentation
```

## Key Conventions

- Write `hypothesis.md` **before** running any experiment
- Fill `results.md` **after** each run
- All metrics go through `ExperimentLogger` → `logs/experiments.jsonl`
- Train/test splits happen **before** poisoning or curriculum ordering (no leakage)
- Config overrides live in `config/experiments.yaml`, not hardcoded in `run.py`

## CI

GitHub Actions: ruff lint, pytest (coverage ≥ 70%), Docker test suite, smoke imports.

## License

MIT
