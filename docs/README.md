# Documentation Index

Welcome to the **Quantum-Inspired Micro ML Lab** documentation.

## Guides

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Install, Docker setup, run your first experiment |
| [Architecture](architecture.md) | Code structure, module responsibilities, data flow |
| [Experiments](experiments.md) | All 7 experiments — goals, metrics, how to run |
| [Hypothesis Workflow](hypothesis-workflow.md) | Mandatory hypothesis-first discipline |
| [Testing](testing.md) | Test pyramid, coverage thresholds, CI pipeline |
| [Docker](docker.md) | Container services, Makefile targets, troubleshooting |

## Quick Reference

```bash
make docker-build    # Build all images
make test            # Run full test suite in Docker
make experiment      # Run EXP 001
make dashboard       # Start Streamlit on :8501
make lint            # Run ruff linter
```

## Conventions

- **Language:** All code, comments, docs, and commit messages are in English
- **Logs:** `logs/experiments.jsonl` is append-only — never delete entries
- **Hypothesis:** Write `hypothesis.md` before every `run.py` execution
- **Results:** Fill `results.md` after every experiment completes
