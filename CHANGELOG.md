# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-06-17

### Added

- Phase 9: **Nano Trainer** — productized mini training on real datasets
- `src/application/model_registry.py`, `train_nanomodel.py`, `nanotrainer_config.py`
- `src/shared/result.py` — lightweight `Ok`/`Fail` Result type
- `config/nanotrainer.yaml` — model × dataset pairs and `mini`/`ci` profiles
- CLI `qml-train` (`scripts/nano_train.py`) + `make train-demo`
- Streamlit page `dashboard/pages/01_nano_trainer.py`
- `experiments/exp_019_nanotrainer_smoke` — validates all registry models via app path
- `docs/nanotrainer.md` — usage and architecture
- Golden CI bounds for `nano_train` exp_id; leaderboard excludes app runs

### Changed

- Documentation updated to 19 experiments
- Default Nano Trainer profile: `mini` (100 samples, 8 epochs)

## [0.7.0] - 2026-06-17

### Added

- Phase 8: `exp_018_feature_fusion` — Transformer-mini → QNN fusion pipeline
- `sequential_phase` dataset (phase-sensitive, PCA-insufficient temporal task)
- `src/quantum/transformer_qnn_fusion.py` — `TransformerQNNFusion` model
- CI smoke `test_exp_018_smoke.py` + unit tests for dataset and model
- `make fusion` target

### Fixed

- Removed duplicate `exp_017_poison_topology` block in `config/experiments.yaml`

### Changed

- Documentation updated to 18 experiments
- Roadmap complete (all phases 0–8)

## [0.6.0] - 2026-06-17

### Added

- Phase 7: `exp_017_poison_topology` — hybrid topology × label poisoning robustness
- Four arms: sandwich, quantum-first, classical-first, EXP 016 NAS preset
- CI smoke `test_exp_017_smoke.py` + golden bounds
- `make poison-topology` target

### Changed

- Documentation updated to 17 experiments
- `docs/literature_review.md` marks exp_017 complete

## [0.5.0] - 2026-06-17

### Added

- Phase 6: `exp_016_hybrid_nas` — Optuna NAS over hybrid classical–quantum layouts
- `evaluate_hybrid_trial`, `build_hybrid_from_params`, `build_exp_016_objective` in `hpo.py`
- CI smoke `test_exp_016_smoke.py` + golden bounds
- `make nas` target; `run_hpo.py` supports exp_016
- `hpo_trials: 3` in ci profile for fast NAS smoke

### Changed

- Documentation updated to 16 experiments
- `docs/literature_review.md` marks exp_016 complete

## [0.4.1] - 2026-06-17

### Added

- Post-roadmap closure: publication runs for exp_011–exp_015
- `src/training/results_writer.py` and `scripts/generate_results_md.py` (`make results-new`)
- `scripts/run_exp_011_015.py` (`make experiments-new`)
- `results.md` for exp_011–exp_015 (publication profile, 10 seeds)
- Integration smoke `test_exp_011_smoke.py` + golden bounds for UCI perceptron
- `RELEASE_NOTES.md` for v0.4.0 Zenodo release
- `run_exp_011_ci` in `ci_smoke.py`

### Changed

- `docs/zenodo.md` updated for v0.4.0 release workflow
- `prepare_release.py` references v0.4.0 tag
- Roadmap archived and removed (all phases complete)

## [0.4.0] - 2026-06-17

### Added

- Phase 5 engineering excellence
- `pip install -e .` packaging via hatchling (`pyproject.toml` build-system)
- `qml-run` CLI (`scripts/cli.py`) — run experiments with `--profile`
- `make check` — ruff + mypy + pytest (80% cov) + integration + contracts
- `make health` — pre-flight disk/logs/MLflow checks (`scripts/health_check.py`)
- `.pre-commit-config.yaml` — ruff, mypy, hypothesis placeholder guard
- `tests/contracts/` — JSONL schema validation with jsonschema
- CI jobs: mypy, pip-audit, editable install, contracts in experiment-smoke
- `.github/dependabot.yml` — weekly pip and GitHub Actions updates
- `set_experiment_context()` in `structured_log.py` — experimentId/seed/profile on every log line

### Changed

- Coverage threshold raised to **80%** (pyproject.toml, CI, Dockerfile)
- `requirements-dev.txt` adds mypy, jsonschema, pre-commit, pip-audit, hatchling

## [0.3.0] - 2026-06-17

### Added

- Phase 4 innovation: gradient-variance adaptive learning rate
- `src/training/adaptive_lr.py` — `AdaptiveLRConfig`, `train_model_adaptive`
- `train_with_holdout_adaptive` in `holdout.py`
- `cohens_d_paired` effect size in `statistics.py` (paired comparisons)
- Experiment `exp_015_adaptive_qnn` with hypothesis and ablation plan
- `docs/literature_review.md` — barren plateau and adaptive LR context

### Changed

- `compare_conditions_batch` logs Cohen's d per comparison
- Documentation updated to 15 experiments

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
