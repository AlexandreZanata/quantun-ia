# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
