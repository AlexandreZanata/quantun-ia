# Quantum-Inspired Micro ML Lab

[![CI](https://github.com/AlexandreZanata/quantun-ia/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexandreZanata/quantun-ia/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](https://github.com/AlexandreZanata/quantun-ia/actions/workflows/ci.yml)

A laboratory for classical and quantum ML experiments with rigorous measurement, append-only logging, and a retro benchmark dashboard.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone git@github.com:AlexandreZanata/quantun-ia.git
cd quantun-ia

# 2. Local environment (recommended for development)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 3. Verify setup
pytest tests/ -v

# 4. Write hypothesis, then run an experiment
vim experiments/exp_001_quantum_vs_classical/hypothesis.md
python experiments/exp_001_quantum_vs_classical/run.py

# 5. View benchmarks (terminal + browser)
make dashboard-local
# → http://localhost:8501
```

## Docker (alternative)

```bash
make docker-build
make test
make experiment      # runs EXP 001
make dashboard       # Streamlit on :8501
```

## Run All Experiments

```bash
source .venv/bin/activate
for exp in experiments/exp_*/run.py; do python "$exp"; done
python dashboard/terminal_report.py   # ASCII leaderboard
make dashboard-local                  # interactive charts
```

Results append to `logs/experiments.jsonl` (never delete — append only).

## Experiments

| ID | Name | Focus |
|----|------|-------|
| 001 | Quantum vs Classical | QNN vs MLP baselines |
| 002 | Hybrid Architecture | Sandwich / QuantumFirst / ClassicalFirst |
| 003 | Entanglement Effect | Topology ablation (re-upload QNN) |
| 004 | Data Poisoning | Angle vs amplitude encoding under label noise |
| 005 | Curriculum Quantum | Staged easy→hard vs random order |
| 006 | Barren Plateau | Gradient variance vs qubit count |
| 007 | Self-Play Quantum | Hard-example fine-tuning loop |
| 008 | Data Re-upload | Re-upload vs basic QNN vs param-matched classical |
| 009 | Entanglement Basic | Topology ablation (basic QNN, no re-upload) |
| 010 | Poison Re-upload Ablation | Layer depth and LR under poisoning |
| 011 | UCI Tabular QML | Perceptron, MLP, QNN on breast cancer |
| 012 | MNIST PCA QML | Angle vs amplitude on PCA-reduced MNIST |
| 013 | Augmentation Robustness | Gaussian augmentation on noisy circles |
| 014 | Sequence Baselines | RNN, Transformer-mini vs flattened QNN |
| 015 | Adaptive QNN | Gradient-variance LR vs fixed LR (Phase 4) |
| 016 | Hybrid NAS | Optuna search over hybrid layouts (Phase 6) |
| 017 | Poison × Topology | Hybrid layout vs label poisoning (Phase 7) |
| 018 | Feature Fusion | Transformer → QNN on phase sequences (Phase 8) |
| 019 | Nano Trainer Smoke | Validates app path for all registry models (Phase 9) |
| 020 | API Smoke | REST API + SQLite job persistence (Phase 10) |

See [Experiments](docs/experiments.md) for full details.

## Nano Trainer

Run mini training on real data without editing experiment folders:

```bash
make install
qml-train --model perceptron --dataset breast_cancer --profile mini
make train-demo    # CI-profile demo run
make dashboard-local   # Streamlit → Nano Trainer page
```

See [Nano Trainer](docs/nanotrainer.md) for supported model × dataset pairs.

## REST API

```bash
make api              # FastAPI on :8000
make api-demo         # exp_020 smoke
curl http://127.0.0.1:8000/pwa/   # mobile benchmark PWA
```

See [API](docs/api.md) for endpoints and multitenancy headers.

## Reproducibility, HPO & Publication

```bash
make check              # lint + mypy + pytest (80% cov) + contracts
make install            # pip install -e .
make health             # pre-flight checks before large runs
make repro              # CI smoke profile + golden bounds
make reviewer-repro     # artifact evaluator fast path (< 15 min)
make hpo                # Optuna hyperparameter search (exp_011 default)
make figures            # PDF figures from logs/experiments.jsonl
make latex-tables       # LaTeX tables for paper/
make nas                  # Optuna NAS + holdout (exp_016)
make poison-topology      # hybrid × poisoning (exp_017)
make fusion               # Transformer → QNN (exp_018)
make train-demo           # Nano Trainer CI demo (exp_019 path)
make api                  # REST API server (exp_020 path)
make api-demo             # API smoke test
make experiments-new    # publication runs exp_011–015
make results-new        # generate results.md from JSONL summaries (exp_011–018)
make power-analysis     # minimum detectable Cohen's d table by seed count
make release            # Bundle artifacts for Zenodo (SHA-256 manifest)
make release-check      # Verify dist/release MANIFEST.txt checksums
make citation-ready     # Version alignment before Zenodo tag
make paper-build        # LaTeX paper draft (figures + tables + PDF)
make arxiv-bundle       # arXiv upload tarball (see docs/arxiv.md)
make replay-publication-artifacts  # export CSV, figures, LaTeX from logs
make replay-publication            # publication_large + full export pipeline
```

See [Reproducibility](docs/reproducibility.md), [Negative Results](docs/negative_results.md), and [Zenodo guide](docs/zenodo.md).
Paper draft skeleton: `paper/main.tex`.

## Documentation

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/getting-started.md) | Full setup, Makefile, workflow |
| [Experiments](docs/experiments.md) | All 19 experiments + ablations |
| [Nano Trainer](docs/nanotrainer.md) | CLI + Streamlit mini training app |
| [API](docs/api.md) | REST API + benchmark PWA |
| [Literature Review](docs/literature_review.md) | Phase 4 research context |
| [Method: Adaptive LR](docs/method_adaptive_lr.md) | GV-ALR algorithm, config, exp_015 linkage |
| [Negative Results](docs/negative_results.md) | Honest failures (curriculum, self-play, entanglement) |
| [Reproducibility](docs/reproducibility.md) | NeurIPS-style checklist |
| [DVC Remote](docs/dvc_remote.md) | Artifact storage and `dvc push` setup |
| [Hypothesis Workflow](docs/hypothesis-workflow.md) | Mandatory hypothesis-first discipline |
| [Architecture](docs/architecture.md) | Code structure and data flow |
| [Testing](docs/testing.md) | pytest, coverage, CI |
| [Docker](docs/docker.md) | Container reference |
| [Contributing](CONTRIBUTING.md) | PR checklist and conventions |

## Project Structure

```
quantun-ia/
├── src/              # Models, data, training utilities
├── experiments/      # exp_001 – exp_021 + template
├── config/           # experiments.yaml, nanotrainer.yaml
├── dashboard/        # Streamlit monitor + static PWA
├── logs/             # experiments.jsonl (append-only)
├── tests/            # Unit + smoke tests
└── docs/             # Full documentation
```

## Key Conventions

- Write `hypothesis.md` **before** running any experiment
- Fill `results.md` **after** each run with holdout metrics and conclusions
- All metrics go through `ExperimentLogger` → `logs/experiments.jsonl`
- **Holdout eval:** 30% test split before training (all classification experiments)
- **Multi-seed:** 10 seeds in `config/experiments.yaml` (`publication` profile)
- Config overrides live in `config/experiments.yaml`, not hardcoded in `run.py`

## Citation

If you use this software in research, please cite it using [CITATION.cff](CITATION.cff).

**Zenodo DOI (after release):**

See [docs/citation_loop.md](docs/citation_loop.md) for the unified checklist.

1. `make citation-ready && make release`
2. Tag and push; copy DOI into `CITATION.cff`
3. Validate: `pytest tests/contracts/test_citation_cff.py -v`

## CI

GitHub Actions: ruff lint, mypy, pytest (coverage ≥ 80%), integration/contracts smoke, e2e API tests, weekly cron, paper-build (optional).

## License

MIT — see [LICENSE](LICENSE).
