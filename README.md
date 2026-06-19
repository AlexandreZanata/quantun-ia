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
| 021 | QML Backend Parity | `default.qubit` vs `lightning.qubit` on breast cancer QNN |
| 022 | Nano Quantum Parity | Hybrid sandwich vs param-matched classical on UCI tabular |
| 023 | Encoding × Backend | Angle vs amplitude × backends on PCA-MNIST |
| 024 | QuantumNano-BC | Flagship hybrid sandwich vs clinical baselines on breast cancer |
| 025 | Pima Generalization | Cross-dataset parity test on Pima Indians Diabetes |

See [Experiments](docs/experiments.md) and [MicroQML Bench](docs/microqml_bench.md) for full details.

## Public Leaderboard

Hosted MicroQML Bench v1 JSON and viewer (GitHub Pages):

- **Viewer:** https://alexandrezanata.github.io/quantun-ia/leaderboard/
- **JSON:** https://alexandrezanata.github.io/quantun-ia/leaderboard/v1.json

Refresh locally: `make publish-leaderboard` → commits `docs/leaderboard/v1.json`.

## QuantumNano-BC (Flagship Nano Model)

Train the flagship hybrid QML model on real UCI breast cancer data:

```bash
qml-train --model hybrid_sandwich --dataset breast_cancer --profile publication
python experiments/exp_024_quantum_nano_bc/run.py --profile publication
make model-card    # generate model_cards/quantum_nano_bc.md from logs
```

## Nano Trainer

Run mini training on real data without editing experiment folders:

```bash
make install
qml-train --model perceptron --dataset breast_cancer --profile mini
make train-demo    # CI-profile demo run
make dashboard-local   # Streamlit → Nano Trainer page
```

See [Nano Trainer](docs/nanotrainer.md) for supported model × dataset pairs.

## Shippable Nano Models (train → download → run)

Goal: **many real, reproducible nanomodels** — classical and quantum — that anyone can download and run locally. Each model goes through the full pipeline (train, gate, export); **quality over speed** (hours per model is expected).

### One-command ship pipeline

```bash
# Full pipeline: train → gate → publish serve artifact → export → model card
qml-ship --model large_nano_mlp_synthea --profile publication

# Makefile wrapper
make ship MODEL=large_nano_mlp_synthea
make ship MODEL=quantum_nano_bc PROFILE=publication
make ship-all-p0    # Synthea + HIGGS + QuantumNano-BC + calibrated Synthea
```

Stages: `nanomodel_registry.yaml` → train → real gate test → `open_serve` publish → calibration (clinical) → export (ONNX / TorchScript / Hugging Face) → tarball with SHA-256.

### Download and run (end users, no GPU)

```bash
pip install -e .
qml-download --model large_nano_mlp_synthea   # from release / DVC / HF Hub

# CPU inference smoke
python dist/serve_models/large_nano_mlp_synthea/inference/predict.py \
  --input tests/fixtures/synthea_patient_row.json

# Interactive clinic (uses serve checkpoint)
streamlit run dashboard/pages/03_cv_risk_clinic.py

# REST API
make api
curl -X POST http://127.0.0.1:8000/api/v1/predictions ...
```

### Export formats & local platforms

| Format | Best for | Models |
|--------|----------|--------|
| **Native serve bundle** (`best.pt` + `scaler.joblib` + `config.json`) | quantun-ia API, Streamlit | All, including QNN |
| **ONNX** | ONNX Runtime, edge deploy | Classical / MLP only |
| **TorchScript** | PyTorch-only runners | All PyTorch architectures |
| **Hugging Face Hub** | Community download | All shippable models |
| **LM Studio (adjacent)** | Local sidecar via ONNX + schema JSON | Tabular classical — not chat LLMs |

Quantum hybrid models require PennyLane for inference; they ship as **native bundles**, not ONNX.

### Shippable model catalog (target)

| Registry key | Architecture | Dataset | Status |
|--------------|--------------|---------|--------|
| `large_nano_mlp_synthea` | LargeNanoMLP (~1.17M) | Synthea CV | train ✅ · ship ✅ |
| `large_nano_mlp_synthea_calibrated` | LargeNanoMLP + isotonic | Synthea CV | ship ✅ (exp_043) |
| `large_nano_mlp_higgs` | LargeNanoMLP | HIGGS 1M | train ✅ · ship ✅ |
| `quantum_nano_bc` | HybridSandwich 4q | Breast cancer | train ✅ · ship ✅ |
| `large_nano_hybrid_higgs` | Frozen MLP + QNN head | HIGGS | exp_037 · ship 🔲 |
| `quantum_nano_champion` | Fused Q training recipe | Multi-bench | exp_058 · ship 🔲 |

Internal playbook: `.local/RESEARCH_ROADMAP.md` → **Phase Q** (quantum training hypotheses).

### Quantum training hypotheses → shippable models

Novel quantum **training methods** (warm-start, entanglement schedule, GV-ALR head, noise regularization, re-upload curriculum) are tested on **open benchmarks** (HIGGS, NIHR, GoBug, breast cancer). Only hypotheses that pass pre-registered gates become registry entries and get `qml-ship` — failures go to [Negative Results](docs/negative_results.md).

See [Experiments](docs/experiments.md) exp_051–061 and Phase Q in the internal roadmap.

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
make ship MODEL=...       # train → gate → export shippable nanomodel
make download-model MODEL=...  # install bundle from dist/serve_models/
make ship-all-p0          # ship Synthea + HIGGS + QuantumNano-BC bundles
make api                  # REST API server (exp_020 path)
make api-demo             # API smoke test
make experiments-new    # publication runs exp_011–015
make results-new        # generate results.md from JSONL summaries (exp_011–018)
make power-analysis     # minimum detectable Cohen's d table by seed count
make release            # Bundle artifacts for Zenodo (SHA-256 manifest)
make release-check      # Verify dist/release MANIFEST.txt checksums
make dvc-check            # Validate DVC pipeline stages
make dvc-setup            # Install DVC + configure local remote
make dvc-push             # Push tracked artifacts to remote
make finalize-citation DOI=10.5281/zenodo.XXXXXXX   # After Zenodo moderation
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
| [Experiments](docs/experiments.md) | All 25 experiments + ablations |
| [Nano Trainer](docs/nanotrainer.md) | CLI + Streamlit mini training app |
| [Nano Model Factory](docs/nanomodel_factory.md) | Download, ship, and export shippable models |
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
│   └── application/  # predict, ship, export, human scorers
├── experiments/      # exp_001 – exp_065 + template
├── config/           # experiments.yaml, nanotrainer.yaml, nanomodel_registry.yaml
├── artifacts/        # Training checkpoints (gitignored — ship to release/DVC/HF)
├── dist/serve_models/ # Downloadable shippable bundles
├── dashboard/        # Streamlit monitor + CV Risk Clinic
├── logs/             # experiments.jsonl (append-only)
├── model_cards/      # One card per shippable model
├── tests/            # Unit + smoke + real GPU gates
└── docs/             # Full documentation
```

## Key Conventions

- Write `hypothesis.md` **before** running any experiment
- Fill `results.md` **after** each run with holdout metrics and conclusions
- All metrics go through `ExperimentLogger` → `logs/experiments.jsonl`
- **Holdout eval:** 30% test split before training (all classification experiments)
- **Multi-seed:** 10 seeds in `config/experiments.yaml` (`publication` profile)
- Config overrides live in `config/experiments.yaml`, not hardcoded in `run.py`
- **Shippable models:** every promoted nanomodel needs a registry entry, gate test, model card, and downloadable bundle (see **Shippable Nano Models** above)
- **Quantum training hypotheses:** write `hypothesis.md` first; ship only after MicroQML + serve parity gates pass

## Citation

If you use this software in research, please cite it using [CITATION.cff](CITATION.cff).

**Zenodo DOI (after release):**

See [docs/citation_loop.md](docs/citation_loop.md) for the unified checklist.

1. `make citation-ready && make release`
2. Tag and push; copy DOI into `CITATION.cff`
3. Validate: `pytest tests/contracts/test_citation_cff.py -v`

## Academic paper — step-by-step

Use this workflow to turn lab results into a **real submission** (workshop, journal, or arXiv).
Full arXiv upload details: [docs/arxiv.md](docs/arxiv.md) · LaTeX skeleton: [paper/README.md](paper/README.md).

### 1. Lock the scientific claims

| Claim | Evidence | Artifact |
|-------|----------|----------|
| Holdout-fair QML benchmark | exp_021 / exp_022 multi-seed holdout | `experiments/exp_021_*/results.md` |
| Open-data serve models | HIGGS + Synthea checkpoints | `make model-lab` · `qml-ship` *(planned)* |
| Downloadable nanomodels | Native + ONNX + HF bundles | `qml-download` *(planned)* · GitHub Release |
| Human-interpretable CV ranking | 8 literature cases, Spearman ρ ≥ 0.85 | `make exp-041-publication` |
| Sample-size stability | Precision/AUC curve n=100→2000 | `make exp-042-publication` |

Write `hypothesis.md` **before** each new experiment; fill `results.md` **after** GPU runs.
All metrics must go through `ExperimentLogger` → `logs/experiments.jsonl`.

### 2. Run publication-profile experiments

```bash
source .venv/bin/activate
source .local/env.sh          # CUDA workstation only
make check                    # unit + integration green
make check-real               # RTX 4060 real gates
make exp-041-publication      # clinical case validation
make exp-042-publication      # sample-scale precision curve
make export-model-results     # consolidated JSON (local logs/)
```

Copy headline numbers from:

- `experiments/exp_041_human_cv_clinical_cases/results.md`
- `experiments/exp_042_sample_scale_precision/results.md`

For Synthea CV: report **ROC-AUC @ n=2000** (not n=100 — zero negatives in stratified draw).
For human clinic: report **Spearman ρ** and case ordering, not absolute risk %.

### 3. Build the LaTeX paper

```bash
make paper-build              # latex-tables + figures + sync + pdflatex/bibtex
make arxiv-bundle             # dist/arxiv/quantun-ia-paper.tar.gz
```

| Path | Purpose |
|------|---------|
| `paper/main.tex` | Entry point |
| `paper/sections/` | Intro, methods, experiments, results, limitations |
| `paper/tables/` | Auto-generated from `make latex-tables` |
| `paper/references.bib` | Bibliography (ACC/AHA, Framingham, Synthea) |
| `paper/arxiv_metadata.yaml` | Title, abstract, categories |

Add a **Human validation** subsection citing exp_041 (rank correlation) and exp_042 (sample-scale table).
State clearly: synthetic Synthea cohort, not clinical deployment.

### 4. Pre-submission checklist

- [ ] `make paper-build` succeeds locally and in CI (`paper-build` job)
- [ ] exp_041 + exp_042 `results.md` filled with GPU run dates and hardware
- [ ] Limitations section mentions ~99% label prevalence and ranking vs calibration
- [ ] `CITATION.cff` Zenodo DOI set after release tag ([docs/zenodo.md](docs/zenodo.md))
- [ ] Figures synced: `make figures && make paper-sync`

### 5. Submit

**arXiv:** upload `dist/arxiv/quantun-ia-paper.tar.gz` → categories `cs.LG` + `quant-ph`  
**Journal/workshop:** follow venue template; attach Zenodo archive of code + checkpoints

After moderation, record `arxiv_id` in `paper/arxiv_metadata.yaml` and commit.

### 6. Post-publication

1. Paste DOI / arXiv ID into README and `CITATION.cff`
2. Open a release on GitHub with the Zenodo badge
3. Propose ablations (see exp_041/exp_042 limitations sections)

## CI

GitHub Actions: ruff lint, mypy, pytest (coverage ≥ 80%), integration/contracts smoke, e2e API tests, weekly cron, paper-build (optional).

## License

MIT — see [LICENSE](LICENSE).
