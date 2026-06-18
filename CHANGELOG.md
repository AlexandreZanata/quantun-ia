# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-06-18

### Added

- **Continuous training** — `src/training/champion.py`, `scripts/continuous_train.py`, `exp_027`
- **Chatbot tool** — `src/application/chatbot_tool.py`, `exp_028` (10 golden dialogues)
- **Batch scoring** — `src/application/batch_predict.py`, `scripts/batch_predict.py`, `exp_029`
- **Scale monitoring** — `exp_030` publication_large 30-seed stability gate
- **Clinical curriculum** — `exp_031` margin_batches vs random on breast cancer
- `make phase-v1.1.0-preflight` — RTX 4060 gate before v1.1.0 tag
- `docs/releases/v1.1.0.md` — release evidence for Phases F–J

### Changed

- Version **1.1.0** — continuous training + chatbot + batch application tracks
- `make check-real` expanded to **12/12** real GPU tests on RTX 4060
- `config/nanotrainer.yaml` — `publication_large` profile for weekly challenger runs

## [1.0.0] - 2026-06-18

### Added

- **Real application inference** — `POST /api/v1/predictions`, checkpoint + scaler bundle, `make train-ship`
- `scripts/demo_predict.py` — train QuantumNano-BC and score raw UCI rows on RTX 4060
- `tests/real/test_checkpoint_inference.py` — 7/7 real gate on NVIDIA RTX 4060

### Changed

- Version **1.0.0** — first stable release of the real-application stack (Phases A–D + inference)
- `RELEASE_NOTES.md` and `docs/releases/v1.0.0.md` — v1.0.0 release evidence

## [1.0.0-rc1] - 2026-06-18

### Added

- Phase D (open science): `make phase-d-preflight` — RTX 4060 real gate + release bundle + leaderboard
- `docs/releases/v1.0.0-rc1.md` — release tag evidence for Phase D closure
- Release bundle includes exp_025, exp_026 results and `docs/testing.md`

### Changed

- Version bump to **1.0.0-rc1** — first release candidate for real-application stack
- `RELEASE_NOTES.md` — Phases B–D highlights (real GPU gate, exp_026, publication refresh)
- `docs/citation_loop.md`, `docs/zenodo.md`, `docs/reviewer_guide.md` — v1.0.0-rc1 target

### Includes (since v0.9.22)

- Phase B: `make check-real`, exp_026 API=CLI parity, `set_global_seed` fix
- Phase C: `make phase-c-publication`, `test_publication_bounds.py`, compute_environment wall-clock

## [0.9.23] - 2026-06-18

### Added

- Phase C (publication refresh): `make exp-024-publication`, `make exp-025-publication`, `make phase-c-publication` — RTX 4060 local gate
- `tests/real/test_publication_bounds.py` — 10-seed hybrid mean within exp_024/025 publication 95% CI
- `docs/testing.md` — Tier 1 (CI) vs Tier 2 (`make check-real`) real GPU validation
- `docs/research_agenda.md` — exp_026 real-app E2E row (Q3 2026 track)

### Changed

- Phase C complete: exp_024/025 publication re-run on RTX 4060 (30 seeds, ~65s / ~58s; combined 2m 3s)
- `experiments/exp_024_quantum_nano_bc/results.md` and `experiments/exp_025_pima_generalization/results.md` refreshed
- `model_cards/quantum_nano_bc.md` regenerated from latest publication JSONL
- `docs/compute_environment.md` — Intel i7-13620H, 32 GB RAM, Phase C wall-clock benchmarks

## [0.9.22] - 2026-06-18

### Added

- Phase F (generalization): **exp_025 Pima Indians Diabetes** — cross-dataset parity vs exp_024
- `load_pima_diabetes_raw()` (OpenML id=37) + `pima_diabetes` dataset registry entry
- `experiments/exp_025_pima_generalization/` with hypothesis + run.py
- `run_exp_025_ci` smoke + `tests/integration/test_exp_025_smoke.py`
- MicroQML Bench task `tabular_pima_diabetes`; Zenodo export for Pima CSV

### Fixed

- `resolve_device(model=...)` forces CPU for PennyLane hybrid models on CUDA hosts
- Holdout/self-play eval align tensors to model device (CUDA-safe)

### Changed

- Phase F publication complete: exp_025 Pima parity verdict (Δ=−1.0 pp, 30 seeds, RTX 4060)
- `tests/conftest.py` auto-sets `QML_DEVICE=cuda` when NVIDIA GPU available
- `docs/compute_environment.md` records RTX 4060 Laptop GPU for publication runs
- `paper/tables/exp_025_holdout.tex` from publication JSONL

## [0.9.21] - 2026-06-18

### Added

- Phase E (open science release): **`make open-science-preflight`** — Zenodo/arXiv readiness pipeline
- `scripts/open_science_preflight.py` + `tests/contracts/test_open_science_preflight.py`
- `AUTHORS.md` — author attribution for citation and arXiv
- Phase 30 (flagship): **exp_024 QuantumNano-BC** — hybrid sandwich vs logistic regression, XGBoost, perceptron on full breast cancer
- Phase 31 (publication package): weekly `repro-publication` CI workflow, `make paper-build-publication`, `scripts/repro_publication_ci.sh`
- Phase 32 (external adoption): public MicroQML leaderboard on GitHub Pages (`docs/leaderboard/`)
- `scripts/publish_leaderboard.py` + `make publish-leaderboard` / `publish-leaderboard-check`
- `.github/workflows/pages.yml` — deploy leaderboard viewer + JSON on push to main
- `tests/contracts/test_public_leaderboard.py` — public leaderboard contract tests
- `tests/contracts/fixtures/publication_experiments.jsonl` — publication-profile JSONL for paper-build (exp_021–024)
- `tests/contracts/test_publication_repro.py` — publication pipeline contract tests
- `src/classical/sklearn_wrapper.py`, `logistic_baseline.py`, `xgboost_baseline.py` — clinical tabular baselines
- `scripts/generate_model_card.py` + `model_cards/quantum_nano_bc.md` — flagship model card
- `config/experiments.yaml` exp_024 with 30-seed publication profile and `profile_overrides`
- `config/nanotrainer.yaml` `publication` profile (50 epochs, checkpoints, full dataset)
- MicroQML Bench v1 flagship task `quantum_nano_bc`
- `tests/integration/test_exp_024_smoke.py`, `tests/unit/test_sklearn_classifier.py`, `tests/unit/test_generate_model_card.py`
- DVC `model_card` stage; `make model-card` target

### Changed

- `scripts/prepare_release.py` — seeds publication JSONL; bundles leaderboard JSON + model card + exp_024 results
- `src/training/holdout.py` — sklearn model training path via `SklearnBinaryClassifier`
- `src/training/config.py` — merge `profile_overrides` per experiment profile
- `src/application/train_nanomodel.py` — publication profile saves checkpoints
- `docs/experiments.md`, `docs/nanotrainer.md`, `README.md` — QuantumNano-BC flagship docs
- `docs/compute_environment.md` — exp_024 hardware traceability, 30-seed publication profile
- `docs/microqml_bench.md`, `README.md` — public GitHub Pages leaderboard URLs
- `docs/citation_loop.md`, `docs/zenodo.md` — v0.9.21 open-science workflow
- `CONTRIBUTING.md` — "Reproduce QuantumNano-BC in 15 minutes" section
- `paper/sections/results.tex` — exp_024 flagship headline
- `.github/workflows/release.yml` — publication fixture for reproducible Zenodo bundle

### Dependencies

- `xgboost>=2.0.0`, `joblib>=1.3.0` in `requirements.txt`

## [0.9.20] - 2026-06-17

### Added

- Phase 29 (foundation): **Zenodo reference dataset export** — `scripts/export_reference_datasets.py`, `make export-reference-datasets`
- `experiments/exp_019_nanotrainer_smoke/results.md` and `exp_020_api_smoke/results.md` — infrastructure experiment documentation
- `experiments/exp_023_encoding_backend/results.md` — publication-profile encoding×backend results
- `config/experiments.yaml` entries for exp_019 and exp_020 (`infrastructure: true`)

### Changed

- `README.md` and `docs/experiments.md` — document experiments 019–023
- `scripts/prepare_release.py` — bundles `reference_datasets/` and exp_023 results
- `docs/research_agenda.md` — exp_023 status updated to partial publication result
- `src/application/train_nanomodel.py` — use actual `X_train.shape[1]` for model input (fixes MNIST PCA)
- `src/data/real_datasets.py` — `n_features` metadata after PCA reflects component count
- `experiments/exp_023_encoding_backend/run.py` — skip paired comparisons with unequal seed coverage

### Fixed

- Nano Trainer `quantum_amplitude` × `mnist_binary` shape mismatch (784 vs PCA-8) in exp_019 smoke
- Phase 29: **DVC remote bootstrap** — `make dvc-setup`, `make dvc-push`, `make dvc-push-full`
- `scripts/dvc_remote_setup.py`, `scripts/dvc_push.py`, `dvc>=3.0.0` in `requirements-dev.txt`

### Changed (continued)

- `scripts/validate_dvc.py` — detects `python -m dvc` in venv
- `docs/dvc_remote.md` — documents Make targets (no system `dvc` required)

## [0.9.19] - 2026-06-17

### Added

- Phase 28: **exp_023 encoding×backend interaction** — PCA-MNIST 2×2 factorial (angle/amplitude × default/lightning)
- `experiments/exp_023_encoding_backend/` — hypothesis + run.py
- `run_exp_023_ci` + `tests/integration/test_exp_023_smoke.py`
- `QuantumNetAmplitude` — optional `qml_device` for PennyLane backend parity with angle QNN

### Changed

- `config/experiments.yaml` — exp_023 entry with ci/publication profiles
- `docs/research_agenda.md` — exp_023 status CI smoke
- `tests/regression/golden_ci.json` — exp_023 bounds

## [0.9.18] - 2026-06-17

### Added

- Phase 27: **DVC artifact validation** — `make dvc-check`, `scripts/validate_dvc.py`
- Contract tests: `tests/contracts/test_dvc_pipeline.py`
- Unit tests: `tests/unit/test_validate_dvc.py`

### Changed

- `scripts/health_check.py` — DVC check uses pipeline validator
- `docs/dvc_remote.md` — documents `make dvc-check`

## [0.9.17] - 2026-06-17

### Added

- Phase 26: **Post-tag citation closure** — `make finalize-citation`, `docs/releases/v0.9.16.md`
- `scripts/finalize_citation.py` — apply Zenodo DOI + arXiv ID to CITATION, bib, metadata
- Contract tests: `tests/contracts/test_release_record.py`
- Unit tests: `tests/unit/test_finalize_citation.py`

### Changed

- `docs/citation_loop.md` — documents `finalize-citation` after tag `v0.9.16` push
- Release bundle includes `docs/releases/v0.9.16.md`

## [0.9.16] - 2026-06-17

### Added

- Phase 25: **Citation loop readiness** — `make citation-ready`, `docs/citation_loop.md`
- `scripts/validate_citation_ready.py` — version alignment across CITATION, arXiv, release
- `docs/paper_narrative.md` — locked Option C scope for v1 paper
- Contract tests: `test_citation_loop.py`, `test_paper_narrative.py`
- Unit tests: `test_validate_citation_ready.py`

### Changed

- P0 narrative locked — paper contracts enforce headline vs deferred experiments
- Release bundle includes citation loop and narrative docs

## [0.9.15] - 2026-06-17

### Added

- Phase 24: **Method documentation** — `docs/method_adaptive_lr.md` (GV-ALR algorithm, pseudocode, exp_015 linkage)
- Contract tests: `tests/contracts/test_method_docs.py`

### Changed

- `docs/baselines.md`, `docs/architecture.md`, root `README.md` link to method doc
- Release bundle includes `method_adaptive_lr.md`

## [0.9.14] - 2026-06-17

### Added

- Phase 23: **Collaboration & artifact evaluation** — `CODEOWNERS`, GitHub issue templates, `make reviewer-repro`
- `docs/reviewer_guide.md` — ACM/NeurIPS reviewer fast path and author-run statement
- `scripts/reviewer_repro.sh` — one-click reproduction for artifact evaluators
- Contract tests: `tests/contracts/test_collaboration_artifacts.py`
- CI and coverage badges in root `README.md`

### Changed

- `docs/reproducibility.md` — author-run statement, updated reviewer checklist (80% cov)
- `CONTRIBUTING.md` — replication challenge section
- Release bundle includes `reviewer_repro.sh`, `reviewer_guide.md`, `reproducibility.md`

## [0.9.13] - 2026-06-17

### Added

- Phase 19b: **arXiv submission pipeline** — `paper/arxiv_metadata.yaml`, `make arxiv-bundle`, `docs/arxiv.md`
- Publication-frozen holdout tables for exp\_021 and exp\_022 in `paper/tables/`
- `scripts/prepare_arxiv_submission.py` with `00README.txt` and tarball export
- Contract tests: `tests/contracts/test_arxiv_metadata.py`
- Unit tests: `tests/unit/test_prepare_arxiv_submission.py`

### Changed

- Paper `results.tex` embeds headline tables and figures (cross-experiment, exp\_011)
- `paper-build` CI job also runs `arxiv-bundle-sources`
- Experiment suite table extended with exp\_021 and exp\_022

## [0.9.12] - 2026-06-17

### Added

- Phase 18b: **Zenodo citation loop ready** — release bundle v0.9.12, `SECURITY.md`, weekly CI cron
- Release bundle now includes compliance docs, exp_021/022 `results.md`, MicroQML Bench JSON
- `RELEASE_NOTES.md` updated for Phases 0–22b

### Changed

- `docs/zenodo.md` and root `README.md` citation section aligned to v0.9.12 tag workflow
- `prepare_release.py` ships `SECURITY.md`, ethics/compute docs, publication results

## [0.9.11] - 2026-06-17

### Added

- Phase 22b: **exp_022 publication profile** — 10-seed run on breast_cancer + wine_binary + `results.md`
- `src/application/parity_results_writer.py` — uniform results.md for Nano Parity Bench
- `make nano-parity-publication` + `--profile` / `QML_PROFILE` on exp_022 `run.py`
- `tests/unit/test_parity_results_writer.py`

### Changed

- exp_022 `run.py` auto-writes `results.md` on publication profile runs
- `test_results_md_uniform.py` includes exp_022

## [0.9.10] - 2026-06-17

### Added

- Phase 18 (partial): **citation & compliance** — `docs/compute_environment.md`, `docs/ethics.md`, ORCID in `CITATION.cff`
- OSF pre-registration contract — `tests/contracts/test_osf_prereg.py` for exp_021 + exp_022
- Phase 19: **exp_021 publication verdict** — 10-seed `publication` profile run + `results.md`
- `exp_021` in `generate_results_md.py` and `test_results_md_uniform.py`

### Changed

- `hypothesis.md` for exp_021 and exp_022 now include OSF pre-registration URLs

## [0.9.9] - 2026-06-17

### Added

- Phase 20: **e2e in CI** + **`golden_publication.json`** publication-profile regression
- `src/training/publication_smoke.py` — fast 2-seed publication smokes for exp_011 + exp_021
- `tests/regression/golden_publication.json` — narrow holdout bounds for publication drift detection
- `tests/integration/test_publication_golden.py` + `tests/unit/test_publication_smoke.py`
- Dedicated `e2e` job in `.github/workflows/ci.yml` + `make e2e`

## [0.9.8] - 2026-06-17

### Added

- Phase 22: **Nano Parity Bench** + **exp_022** — fair quantum nanomodel vs parameter-matched classical MLP
- `qml-bench-parity` CLI — dataset download, multi-seed holdout, Wilcoxon + Cohen's d verdict
- `config/nano_parity_bench.yaml` — CI/publication profiles, suite, primary claim
- `src/application/nano_parity_bench.py`, `parity_datasets.py`, `parity_config.py`
- `experiments/exp_022_nano_quantum_parity/` — hypothesis + `run.py`
- `make nano-parity-bench`, `make nano-parity-download`
- `run_exp_022_ci` + `tests/integration/test_exp_022_smoke.py` + `tests/unit/test_nano_parity_bench.py`
- `golden_ci.json` bounds for `exp_022.hybrid_sandwich_breast_cancer`

## [0.9.7] - 2026-06-17

### Added

- Phase 17: **MicroQML Bench v1** — versioned external benchmark release
- `config/microqml_bench/v1.yaml` — four primary tasks (synthetic, tabular, PCA-MNIST, sequence)
- `src/benchmark/microqml_bench.py` + `scripts/export_microqml_bench.py` + `make microqml-bench`
- `GET /api/v1/benchmarks/microqml/v1` — schema v1 JSON export
- `tests/contracts/microqml_bench_schema.py` + contract tests
- `docs/microqml_bench.md` — citation, reproduction, versioning policy

## [0.9.6] - 2026-06-17

### Added

- Phase 16: **JWT RS256 auth** — `/api/v1/auth/token` and `/api/v1/auth/refresh` with rotation
- **Async training job queue** — `async_mode` returns `202 PENDING`; background `TrainingJobWorker`
- `device` field on training jobs (`auto` | `cpu` | `cuda`)
- `refresh_tokens` SQLite table + `src/infrastructure/auth/` module
- `tests/e2e/test_api_auth_async.py` — JWT + async job e2e coverage
- `pyjwt[crypto]` dependency

### Changed

- `create_training_job` delegates execution to `process_training_job`
- Protected routes accept Bearer JWT; `X-Tenant-ID` fallback when `API_AUTH_REQUIRED=0`
- `docs/api.md` documents auth env vars and async polling

## [0.9.5] - 2026-06-17

### Added

- Phase 15: **`docs/research_agenda.md`** — public 12-month falsifiable roadmap
- **`experiments/exp_021_qml_backend_parity`** — PennyLane `default.qubit` vs `lightning.qubit` on breast cancer QNN
- `src/quantum/pennylane_device.py` — configurable PennyLane device resolution
- `run_exp_021_ci` + `tests/integration/test_exp_021_smoke.py`
- `tests/unit/test_pennylane_device.py`

### Changed

- `QuantumNetBasic` / `make_qnn_basic` accept optional `qml_device` parameter
- `config/experiments.yaml` — exp_021 entry with ci/publication profiles

## [0.9.4] - 2026-06-17

### Added

- Phase 14: **Statistical upgrade** — uniform `results.md` with verdict, power analysis, Cohen's d labels
- `src/training/effect_size.py` — magnitude labels and minimum detectable effect (MDE)
- `scripts/power_analysis.py` + `make power-analysis`
- `tests/contracts/test_results_md_uniform.py` for exp_011–018
- Protocol comparison table in `docs/baselines.md`

### Changed

- `results_writer.py` adds Verdict, Power analysis, and formatted Cohen's d columns
- Regenerated `results.md` for exp_011–018 via `make results-new`

## [0.9.3] - 2026-06-17

### Added

- Phase 13: **`make replay-publication`** and **`make replay-publication-artifacts`**
- `scripts/replay_publication.py` — orchestrates publication_large + export pipeline
- `docs/dvc_remote.md` + `.dvc/config.example` — DVC remote setup guide
- `tests/unit/test_replay_publication.py`

### Fixed

- Paper build tolerates pdflatex warnings; CI installs `texlive-bibtex-extra`

### Changed

- `docs/reproducibility.md` documents replay runtime and DVC workflow

## [0.9.2] - 2026-06-17

### Added

- Phase 12: **Paper v1 readiness** — contribution statement, limitations section, `make paper-build`
- `scripts/build_paper.py` — sync figures and compile `paper/main.pdf`
- CI `paper-build` job (LaTeX, `continue-on-error`)
- `tests/unit/test_build_paper.py`

### Fixed

- Pin `pandas==2.2.3` in `requirements.lock` (mlflow requires `pandas<3`; fixes CI/Docker)

### Changed

- Paper sections updated for 20 experiments and Option C narrative
- `requirements.txt` caps pandas below 3.x

## [0.9.1] - 2026-06-17

### Added

- Phase 11: **Zenodo release bundle** — SHA-256 `MANIFEST.txt`, static artifacts, verification CLI
- `make release-check` — verify checksums in `dist/release/`
- `tests/unit/test_prepare_release.py` — manifest and checksum contract tests
- `tests/contracts/test_citation_cff.py` — CITATION.cff version and DOI format validation
- GitHub Actions release workflow publishes bundle assets on tag push

### Changed

- `docs/zenodo.md`, `RELEASE_NOTES.md` updated for v0.9.0 (REST API + PWA)
- Release bundle includes `CITATION.cff`, `CHANGELOG.md`, `docs/api.md`

## [0.9.0] - 2026-06-17

### Added

- Phase 10: **REST API** — FastAPI service wrapping Nano Trainer
- `src/domain/entities/training_job.py` + SQLite `training_jobs` table
- `src/application/create_training_job.py` — persist and run jobs with `tenantId`
- `src/presentation/http/app.py` — `/health`, `/ready`, `/metrics`, training-jobs, leaderboard
- Mobile benchmark **PWA** at `/pwa/`
- CLI `qml-api` + `make api` / `make api-demo`
- `experiments/exp_020_api_smoke` + `tests/e2e/test_api_routes.py`
- `docs/api.md`

### Changed

- Documentation updated to 20 experiments
- `docs/nanotrainer.md` — API surface documented

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
