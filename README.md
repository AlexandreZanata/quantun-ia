# Quantum-Inspired Micro ML Lab

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

See [Experiments](docs/experiments.md) for full details.

## Reproducibility, HPO & Publication

```bash
make check              # lint + mypy + pytest (80% cov) + contracts
make install            # pip install -e .
make health             # pre-flight checks before large runs
make repro              # CI smoke profile + golden bounds
make hpo                # Optuna hyperparameter search (exp_011 default)
make figures            # PDF figures from logs/experiments.jsonl
make latex-tables       # LaTeX tables for paper/
make experiments-new    # publication runs exp_011–015
make results-new        # generate results.md from JSONL summaries
make release            # Bundle artifacts for Zenodo v0.4.0
```

See [Reproducibility](docs/reproducibility.md), [Negative Results](docs/negative_results.md), and [Zenodo guide](docs/zenodo.md).
Paper draft skeleton: `paper/main.tex`.

## Documentation

| Doc | Description |
|-----|-------------|
| [Getting Started](docs/getting-started.md) | Full setup, Makefile, workflow |
| [Experiments](docs/experiments.md) | All 15 experiments + ablations |
| [Literature Review](docs/literature_review.md) | Phase 4 research context |
| [Baselines](docs/baselines.md) | Literature comparison table |
| [Negative Results](docs/negative_results.md) | Honest failures (curriculum, self-play, entanglement) |
| [Reproducibility](docs/reproducibility.md) | NeurIPS-style checklist |
| [Hypothesis Workflow](docs/hypothesis-workflow.md) | Mandatory hypothesis-first discipline |
| [Architecture](docs/architecture.md) | Code structure and data flow |
| [Testing](docs/testing.md) | pytest, coverage, CI |
| [Docker](docs/docker.md) | Container reference |
| [Contributing](CONTRIBUTING.md) | PR checklist and conventions |

## Project Structure

```
quantun-ia/
├── src/              # Models, data, training utilities
├── experiments/      # exp_001 – exp_015 + template
├── config/           # experiments.yaml (central hyperparameters)
├── dashboard/        # Retro Streamlit benchmark monitor
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

## CI

GitHub Actions: ruff lint, pytest (coverage ≥ 70%), Docker test suite, smoke imports.

## License

MIT — see [LICENSE](LICENSE).
