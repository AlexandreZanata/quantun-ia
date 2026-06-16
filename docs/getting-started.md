# Getting Started

## Prerequisites

- Python 3.11+ (3.12 recommended)
- 4 GB RAM minimum (quantum simulators are memory-intensive)
- Docker + Docker Compose (optional, for containerized runs)

## Local Setup (Recommended)

```bash
git clone git@github.com:AlexandreZanata/quantun-ia.git
cd quantun-ia

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU PyTorch (saves disk space)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
pytest tests/ -v --cov=src --cov-fail-under=70
python -c "import pennylane, torch, streamlit; print('OK')"
```

## Run Your First Experiment

1. **Write your hypothesis** in `experiments/exp_001_quantum_vs_classical/hypothesis.md`
2. **Run:**

```bash
python experiments/exp_001_quantum_vs_classical/run.py
```

3. **View results:**

```bash
make dashboard-local
# Terminal leaderboard + browser at http://localhost:8501
# Click [ REFRESH DATA ] after new runs
```

4. **Document** findings in `experiments/exp_001_quantum_vs_classical/results.md`

## Run All Experiments

```bash
source .venv/bin/activate
python experiments/exp_001_quantum_vs_classical/run.py
python experiments/exp_002_hybrid_architecture/run.py
python experiments/exp_003_entanglement_effect/run.py
python experiments/exp_004_data_poisoning/run.py
python experiments/exp_005_curriculum_quantum/run.py
python experiments/exp_006_barren_plateau/run.py
python experiments/exp_007_self_play/run.py
```

Or in one line:

```bash
for f in experiments/exp_*/run.py; do python "$f"; done
```

## Dashboard

```bash
make dashboard-local
```

This runs:
1. `dashboard/terminal_report.py` — ASCII leaderboard in the terminal
2. `streamlit run dashboard/app.py` — retro benchmark UI at **http://localhost:8501**

> `make dashboard-local` uses `.venv/bin/python` automatically when the venv exists.

## Docker Setup

```bash
make docker-build
make test
make experiment
make dashboard    # http://localhost:8501
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make dev` | Interactive dev container |
| `make test` | Full test suite in Docker |
| `make test-watch` | pytest with live output |
| `make lint` / `make lint-fix` | ruff check / auto-fix |
| `make coverage` | HTML coverage report |
| `make dashboard` | Streamlit via Docker |
| `make dashboard-local` | Terminal report + Streamlit (local venv) |
| `make experiment` | Run EXP 001 in Docker |
| `make docker-build` | Build all images |
| `make clean` | Remove `__pycache__`, `.pytest_cache` |

## Configuration

Hyperparameters live in `config/experiments.yaml`. Experiments load them via:

```python
from src.training.config import load_experiment_config
cfg = load_experiment_config("exp_004_data_poisoning")
```

## Important Rules

- **Hypothesis first** — no `run.py` without a written `hypothesis.md`
- **Append-only logs** — never delete `logs/experiments.jsonl`
- **No data leakage** — train/test split before poisoning, curriculum, or self-play
- **English only** — all code, comments, docs, and commit messages

## Next Steps

- [Experiments](experiments.md) — goals, ablations, and flags
- [Hypothesis Workflow](hypothesis-workflow.md) — rigorous experiment discipline
- [Architecture](architecture.md) — module map and data flow
