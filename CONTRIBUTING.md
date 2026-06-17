# Contributing to quantun-ia

Thank you for contributing to the Quantum-Inspired Micro ML Lab. This project follows
strict research discipline — read this guide before opening a pull request.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Familiarity with PennyLane, PyTorch, and pytest
- All artifacts (code, comments, docs, commit messages) must be in **English**

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Hypothesis-First Rule (mandatory)

Every experiment **must** have a non-placeholder `hypothesis.md` written **before** `run.py` runs.

1. Copy `experiments/template/` to `experiments/exp_NNN_descriptive_name/`
2. Fill `hypothesis.md` with:
   - What you expect to happen
   - Why you expect it
   - What would prove you wrong
   - Metrics you will measure
3. Only then write or modify `run.py`
4. After running, fill `results.md` with holdout metrics and conclusions

CI rejects placeholder hypothesis content via `tests/unit/test_hypothesis.py`.

## Adding a New Experiment

1. Create folder: `experiments/exp_NNN_<name>/`
2. Add entry in `config/experiments.yaml` under `experiments:`
3. Import logic from `src/` — keep `run.py` as a thin orchestrator
4. Log all metrics through `ExperimentLogger` (`src/training/metrics.py`)
5. Use `src/data/splits.py` for stratified splits **before** any preprocessing
6. Add smoke import coverage in `tests/smoke/test_experiment_imports.py` (automatic via parametrize)

## Configuration

- Hyperparameters live in `config/experiments.yaml` — never hardcode in `run.py`
- Use profiles (`publication`, `publication_large`) for dataset size and seed count
- Load config via `load_experiment_config("exp_NNN_<name>")`

## Running Tests

```bash
make lint          # ruff check
pytest tests/ -v   # full suite
make coverage      # HTML report with 70% threshold
```

## Pull Request Checklist

- [ ] `hypothesis.md` exists and has no placeholder text
- [ ] `results.md` updated if experiment was run
- [ ] `config/experiments.yaml` updated if new experiment or hyperparameter change
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Lint passes: `ruff check src/ tests/ experiments/`
- [ ] `logs/experiments.jsonl` was **not** committed (append-only, gitignored)
- [ ] No secrets, tokens, or credentials in code or logs
- [ ] English only in all new artifacts

## Log Discipline

- `logs/experiments.jsonl` is **append-only** — never delete entries
- Never write directly to `logs/` — always use `ExperimentLogger`
- Never log passwords, tokens, or PII

## Code Style

- Python modules: `snake_case.py`
- Experiment folders: `exp_NNN_descriptive_name`
- Git branches: `feat/`, `fix/`, `exp/` prefixes
- Follow existing patterns in `src/` — read surrounding code before adding abstractions

## Questions

Open an issue or refer to:

- [Getting Started](docs/getting-started.md)
- [Hypothesis Workflow](docs/hypothesis-workflow.md)
- [Architecture](docs/architecture.md)
- [Reviewer Guide](docs/reviewer_guide.md) — artifact evaluation and replication

## Replication challenge

Independent replicators are welcome. Fast verification:

```bash
make reviewer-repro
```

For a full publication replay (long), use `make replay-publication`.

When reporting results, open a GitHub issue with the **Experiment Replication Challenge** template
(`.github/ISSUE_TEMPLATE/experiment_replication.yml`). Include experiment ID, profile, commit SHA,
your holdout metrics, and a verdict vs `experiments/exp_NNN_*/results.md`.
