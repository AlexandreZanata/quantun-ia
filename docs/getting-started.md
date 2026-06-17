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

## Experiment Workflow (all 10)

Every experiment follows the same rigor:

1. Write `hypothesis.md` (before running)
2. Run `python experiments/exp_NNN_<name>/run.py`
3. Check `logs/experiments.jsonl` or `make dashboard-local`
4. Fill `results.md` with holdout metrics and conclusions

**Evaluation defaults** (`config/experiments.yaml`, `publication` profile):

| Setting | Value |
|---------|-------|
| `dataset` | circles |
| `noise` | 0.2 |
| `n_samples` | 500 |
| `test_size` | 0.3 (30% holdout) |
| `seeds` | 10 seeds (42, 123, 456, 789, 1024, 1337, 2048, 3001, 4096, 5000) |

Holdout accuracy is logged as `holdout eval` / `multi-seed summary` in structured logs.

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
python experiments/exp_008_data_reupload/run.py
python experiments/exp_009_entanglement_basic/run.py
python experiments/exp_010_poison_reupload_ablation/run.py
```

Or in one line:

```bash
for f in experiments/exp_*/run.py; do python "$f"; done
```

### Publication Large Profile

For narrower confidence intervals (n=1000):

```bash
QML_PROFILE=publication_large python experiments/exp_001_quantum_vs_classical/run.py
# or
make experiment-large
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
| `make experiment-large` | Run all experiments with `publication_large` profile |
| `make repro` | CI-profile integration smoke (exp_001) |
| `make export-results` | Export JSONL logs to CSV |
| `make hpo` | Run Optuna HPO (default: exp_011, 5 trials) |
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

## GPU (optional)

Classical models auto-detect CUDA when available (`QML_DEVICE=auto` default).
Quantum PennyLane simulators may remain CPU-bound depending on backend.

```bash
# Local: use GPU if available
export QML_DEVICE=cuda
python experiments/exp_011_uci_tabular_qml/run.py

# Docker with NVIDIA runtime
docker compose -f docker-compose.gpu.yml run --rm experiment-gpu
```

## Next Steps

- [Experiments](experiments.md) — goals, ablations, and flags
- [Hypothesis Workflow](hypothesis-workflow.md) — rigorous experiment discipline
- [Architecture](architecture.md) — module map and data flow
- [Contributing](../CONTRIBUTING.md) — PR checklist
