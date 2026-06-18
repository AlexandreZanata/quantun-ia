# Compute Environment

**Lab:** Quantum-Inspired Micro ML Lab (`quantun-ia`)  
**Last updated:** 2026-06-18  
**Software version:** v1.0.0

This document records the hardware and software stack used for **publication-profile** experiment numbers cited in the paper and `results.md` files.

---

## Reference machine (publication runs)

| Field | Value |
|-------|-------|
| CPU | **Intel Core i7-13620H** (16 threads, up to 4.9 GHz) |
| GPU | **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB VRAM, CUDA 13.0, driver 580.159) |
| RAM | **32 GB** |
| OS | Pop!_OS (Linux kernel 7.x) |
| Python | 3.12 |
| PyTorch | ≥ 2.2 (`QML_DEVICE=cuda` for classical heads) |
| PennyLane | ≥ 0.35 (`default.qubit` — CPU simulator for hybrid quantum blocks) |
| XGBoost | ≥ 2.0 (exp_024 clinical baselines) |

### Wall-clock (Phase C refresh — 2026-06-18, this machine)

| Experiment | Profile | Seeds | Models | Wall-clock |
|------------|---------|-------|--------|------------|
| exp_024 QuantumNano-BC | publication | 30 | 5 | **51 s** |
| exp_025 Pima generalization | publication | 30 | 5 | **61 s** |
| **Combined** (`make phase-c-publication`) | publication | 30 each | 5 each | **~2 min** |

Classical PyTorch layers run on **RTX 4060**; PennyLane quantum simulation stays on CPU.

Publication-profile runs (`config/experiments.yaml` → `profile: publication`) use:

- **30 stratified holdout seeds** for flagship exp_024 (10 seeds for exp_021–023 legacy profile)
- 50 training epochs (unless overridden per experiment)
- 30% holdout, preprocessing fit on train only

---

## Flagship experiment (exp_024 QuantumNano-BC)

| Field | Value |
|-------|-------|
| Dataset | Wisconsin Breast Cancer (569 samples, 30 features) |
| Seeds | 30 (publication profile in `config/experiments.yaml`) |
| Models | hybrid_sandwich, logistic_regression, xgboost_shallow, perceptron, classical_matched |
| Backend | PennyLane `default.qubit` (simulator-only) |
| Verdict | Parity accepted — hybrid 97.4% vs logistic 97.9% mean holdout (Δ=−0.5 pp) |
| Wall-clock (Phase C, 2026-06-18) | **51 s** (30 seeds × 5 models, RTX 4060 + PennyLane CPU) |

Record wall-clock time and exact CPU model in `results.md` when citing externally.

---

## Generalization experiment (exp_025 Pima Indians Diabetes)

| Field | Value |
|-------|-------|
| Dataset | Pima Indians Diabetes (OpenML id=37, 768 samples, 8 features) |
| Seeds | 30 (publication profile in `config/experiments.yaml`) |
| Models | hybrid_sandwich, logistic_regression, xgboost_shallow, perceptron, classical_matched |
| Classical backend | PyTorch on **NVIDIA RTX 4060** (`QML_DEVICE=cuda`) |
| Quantum backend | PennyLane `default.qubit` (CPU — TorchLayer constraint) |
| Claim | Parity within 2 pp vs logistic regression (generalization vs exp_024) |
| Wall-clock (Phase C, 2026-06-18) | **61 s** (30 seeds × 5 models, RTX 4060 + PennyLane CPU) |

Local publication command:

```bash
QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_025_pima_generalization/run.py --profile publication --write-results
```

---

## CI vs local

| Environment | Purpose | Profile |
|-------------|---------|---------|
| GitHub Actions `ubuntu-latest` | Regression gates (`golden_ci.json`, e2e) | `ci` (2 seeds, 5–15 epochs) |
| Weekly `repro-publication` workflow | Real exp_024 CI + paper-build from JSONL | `ci` + publication fixture merge |
| `golden_publication.json` smoke | Publication drift detection | `publication` (2 seeds, 15 epochs) |
| Maintainer workstation | Full publication verdicts | `publication` (30 seeds for exp_024/025, 50 epochs); **RTX 4060** for classical models |

Docker reproduction: `make docker-test` (validates wiring; not full publication profile).

---

## Reproducing publication numbers

**Fast path (< 15 min, CI profile):**

```bash
source .venv/bin/activate
make health
MLFLOW_DISABLE=1 python experiments/exp_024_quantum_nano_bc/run.py --profile ci
```

**Full flagship verdict (30 seeds, ~2 min on RTX 4060):**

```bash
QML_DEVICE=cuda MLFLOW_DISABLE=1 make phase-c-publication
# or individually:
QML_DEVICE=cuda MLFLOW_DISABLE=1 make exp-024-publication
QML_DEVICE=cuda MLFLOW_DISABLE=1 make exp-025-publication
```

**Paper artifacts from logs:**

```bash
make paper-build-publication   # uses publication JSONL fixture + LaTeX pipeline
make release                   # Zenodo bundle including model_cards/quantum_nano_bc.md
```

---

## Backend notes

- **`default.qubit`:** always available; backprop gradients.
- **`lightning.qubit`:** requires `pennylane-lightning`; adjoint gradients; may need more RAM on deep circuits.
- Async API jobs (`device: cpu|cuda|auto`) use the same holdout protocol as CLI runs.

See also: [reproducibility.md](reproducibility.md), [zenodo.md](zenodo.md).
