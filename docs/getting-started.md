# Getting Started

## Prerequisites

- Docker and Docker Compose (recommended)
- Python 3.11+ (for local development)
- 4 GB RAM minimum (quantum simulators are memory-intensive)

## Docker Setup (Recommended)

```bash
git clone <repo-url>
cd quantun-ia

# Build images
make docker-build

# Run tests to verify everything works
make test

# Run the first experiment
make experiment

# View results in the dashboard
make dashboard
# Open http://localhost:8501
```

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install CPU PyTorch first (optional, saves disk space)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
python -c "import pennylane; print('PennyLane:', pennylane.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import streamlit; print('Streamlit OK')"
pytest tests/smoke/ -v
```

## Run Your First Experiment

1. **Write your hypothesis** in `experiments/exp_001_quantum_vs_classical/hypothesis.md`
2. **Run the experiment:**

```bash
python experiments/exp_001_quantum_vs_classical/run.py
```

3. **View results:**

```bash
streamlit run dashboard/app.py
```

4. **Fill in** `experiments/exp_001_quantum_vs_classical/results.md`

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make dev` | Start interactive dev container |
| `make test` | Run full test suite in Docker |
| `make test-watch` | Run tests with live output |
| `make lint` | Run ruff linter |
| `make coverage` | Run tests with HTML coverage report |
| `make dashboard` | Start Streamlit dashboard |
| `make experiment` | Run EXP 001 |
| `make docker-build` | Build all Docker images |
| `make clean` | Remove cache files |

## Next Steps

- Read [Hypothesis Workflow](hypothesis-workflow.md) before running more experiments
- Explore [Experiments](experiments.md) for the full 7-experiment roadmap
- See [Architecture](architecture.md) to understand the codebase
