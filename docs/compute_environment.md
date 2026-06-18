# Compute Environment

**Lab:** Quantum-Inspired Micro ML Lab (`quantun-ia`)  
**Last updated:** 2026-06-18  
**Software version:** v0.9.20

This document records the hardware and software stack used for **publication-profile** experiment numbers cited in the paper and `results.md` files.

---

## Reference machine (publication runs)

| Field | Value |
|-------|-------|
| CPU | AMD/Intel x86_64 (Pop!_OS 22.04, kernel 6.x) |
| GPU | Optional — CI and default local runs use **CPU-only** PyTorch |
| RAM | ≥ 16 GB recommended for PennyLane `lightning.qubit` |
| Python | 3.11+ (CI: 3.12) |
| PyTorch | ≥ 2.2 (CPU wheel in CI) |
| PennyLane | ≥ 0.35 (`default.qubit`, `lightning.qubit` when installed) |
| XGBoost | ≥ 2.0 (exp_024 clinical baselines) |

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

Record wall-clock time and exact CPU model in `results.md` when citing externally.

---

## CI vs local

| Environment | Purpose | Profile |
|-------------|---------|---------|
| GitHub Actions `ubuntu-latest` | Regression gates (`golden_ci.json`, e2e) | `ci` (2 seeds, 5–15 epochs) |
| Weekly `repro-publication` workflow | Real exp_024 CI + paper-build from JSONL | `ci` + publication fixture merge |
| `golden_publication.json` smoke | Publication drift detection | `publication` (2 seeds, 15 epochs) |
| Maintainer workstation | Full publication verdicts | `publication` (30 seeds for exp_024, 50 epochs) |

Docker reproduction: `make docker-test` (validates wiring; not full publication profile).

---

## Reproducing publication numbers

**Fast path (< 15 min, CI profile):**

```bash
source .venv/bin/activate
make health
MLFLOW_DISABLE=1 python experiments/exp_024_quantum_nano_bc/run.py --profile ci
```

**Full flagship verdict (30 seeds, ~2–4 h CPU):**

```bash
QML_PROFILE=publication MLFLOW_DISABLE=1 python experiments/exp_024_quantum_nano_bc/run.py --profile publication --write-results --write-model-card
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
