# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-17

### Added

- Phase 3 publication and open science pipeline
- `src/training/plot_style.py` — consistent matplotlib/seaborn style
- `scripts/generate_figures.py` — PDF figures from JSONL logs (`make figures`)
- `scripts/export_latex_tables.py` — LaTeX holdout tables (`make latex-tables`)
- `scripts/prepare_release.py` — Zenodo release bundle (`make release`)
- `docs/negative_results.md` — documented failures (exp_005, 007, 003, 009)
- `docs/reproducibility.md` — NeurIPS-style reproducibility checklist
- `docs/zenodo.md` — GitHub/Zenodo DOI integration guide
- `paper/` — LaTeX draft skeleton (main.tex + sections)
- DVC stages for figures and LaTeX tables
- Rewritten `notebooks/analysis.ipynb` with publication leaderboard

### Changed

- `CITATION.cff` bumped to v0.2.0 (DOI placeholder for Zenodo)
- `.gitignore` adds `figures/` and `dist/`

### Added

- Phase 2 scientific scope expansion
- `src/data/real_datasets.py`, `src/data/dataset_registry.py`, `src/data/scaling.py`
- `load_synthetic_raw()` in generators for split-then-scale workflows
- Experiments exp_011–exp_014 (UCI, MNIST PCA, augmentation, sequence baselines)
- `src/training/hpo.py` and `scripts/run_hpo.py` (Optuna)
- `src/training/device.py` — CUDA/CPU auto-detect in trainer
- `docs/baselines.md` — literature comparison reference
- `docker-compose.gpu.yml` for NVIDIA GPU runs
- Makefile target: `make hpo`

### Changed

- `ClassicalNet` and holdout pipeline support variable `input_dim`
- `trainer.py` moves tensors to resolved device (`QML_DEVICE` env)
- `requirements.txt` adds `optuna`

### Added

- Phase 1 reproducibility infrastructure
- `requirements.lock` for pinned runtime dependencies
- `src/training/reproducibility.py` — global seed utility
- `src/training/tracking.py` — optional MLflow dual-write
- `src/training/checkpoints.py` — model checkpoint persistence
- `src/training/ci_smoke.py` — fast exp_001 CI profile runner
- `scripts/export_results.py` — JSONL to CSV export for DVC
- `dvc.yaml` export pipeline and DVC project initialization
- `ci` profile in `config/experiments.yaml` (n=50, 2 seeds, 5 epochs)
- Integration test `tests/integration/test_exp_001_smoke.py` with golden bounds
- CI job `experiment-smoke`
- Makefile targets: `repro`, `export-results`
- `.env.example` for MLflow and profile configuration

### Changed

- `ExperimentLogger` logs `seed`, `profile`, and dual-writes to MLflow when enabled
- `trainer.py` and `holdout.py` accept `seed`, `profile`, and checkpoint saving
- Holdout experiments pass seed/profile through all `run.py` scripts
- Docker and CI install from `requirements.lock`
- `requirements.txt` adds `mlflow`; `requirements-dev.txt` adds `dvc`

### Added (Phase 0)

- `LICENSE` (MIT)
- `CITATION.cff` for software citation
- `CONTRIBUTING.md` with hypothesis-first PR checklist
- `CHANGELOG.md` (this file)
- Smoke import coverage for `qnn_reupload`, `qnn_factory`, `param_match`, `gradients`, `circuit_utils`, `statistics`, `poisoning`

### Changed

- Documentation synced to 10 experiments and 10-seed publication profile
- `docs/architecture.md` updated with full module map and multi-seed holdout data flow
- `docs/experiments.md` expanded with exp_008–exp_010

## [0.1.0] - 2026-06-16

### Added

- 10 quantum/classical ML experiments (`exp_001`–`exp_010`)
- Central config: `config/experiments.yaml` with `publication` and `publication_large` profiles
- Statistical stack: bootstrap CI, paired Wilcoxon, Holm-Bonferroni correction
- Parameter-matched classical baselines (`src/training/param_match.py`)
- Applicability gates for curriculum and self-play (`src/training/protocol.py`)
- Streamlit dashboard and terminal leaderboard
- CI pipeline: ruff, pytest (≥70% coverage), Docker test suite, smoke imports
- Structured JSON logging via loguru (`src/training/structured_log.py`)

[Unreleased]: https://github.com/AlexandreZanata/quantun-ia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AlexandreZanata/quantun-ia/releases/tag/v0.1.0
