# Documentation Index

Welcome to the **Quantum-Inspired Micro ML Lab** documentation.

## Guides

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Install, venv, Docker, dashboard, run all experiments |
| [Architecture](architecture.md) | Code structure, module responsibilities, data flow |
| [Experiments](experiments.md) | All 15 experiments — goals, ablations, known flags |
| [Literature Review](literature_review.md) | Phase 4 barren plateau and adaptive LR context |
| [Baselines](baselines.md) | Literature comparison for real-data experiments |
| [Negative Results](negative_results.md) | Documented honest failures (exp_005, 007, 003, 009) |
| [Reproducibility](reproducibility.md) | NeurIPS-style reproducibility checklist |
| [Zenodo Release](zenodo.md) | DOI archival guide for v0.9.1+ |
| [DVC Remote](dvc_remote.md) | Artifact remote setup and `dvc push` |
| [Hypothesis Workflow](hypothesis-workflow.md) | Mandatory hypothesis-first discipline |
| [Testing](testing.md) | Test pyramid, coverage thresholds, CI pipeline |
| [Docker](docker.md) | Container services, Makefile targets, troubleshooting |

## Quick Reference

```bash
# Local workflow
source .venv/bin/activate
pytest tests/ -v
python experiments/exp_001_quantum_vs_classical/run.py
make dashboard-local          # http://localhost:8501

# Docker workflow
make docker-build && make test && make experiment && make dashboard
```

## Conventions

- **Language:** All code, comments, docs, and commit messages are in English
- **Logs:** `logs/experiments.jsonl` is append-only — never delete entries
- **Hypothesis:** Write `hypothesis.md` before every `run.py` execution
- **Results:** Fill `results.md` after every experiment completes
- **Config:** Hyperparameters in `config/experiments.yaml`, loaded via `src/training/config.py`
- **Splits:** `src/data/splits.py` — stratified train/test before any preprocessing
- **Holdout:** `src/training/holdout.py` — train on train, eval on test, multi-seed summary
- **Seeds:** 10 seeds in `publication` profile for statistical rigor
- **Citation:** Use [CITATION.cff](../CITATION.cff) when referencing this software
