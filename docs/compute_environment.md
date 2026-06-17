# Compute Environment

**Lab:** Quantum-Inspired Micro ML Lab (`quantun-ia`)  
**Last updated:** 2026-06-17  
**Software version:** v0.9.10

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

Publication-profile runs (`config/experiments.yaml` → `profile: publication`) use:

- 10 stratified holdout seeds
- 50 training epochs (unless overridden per experiment)
- 30% holdout, preprocessing fit on train only

---

## CI vs local

| Environment | Purpose | Profile |
|-------------|---------|---------|
| GitHub Actions `ubuntu-latest` | Regression gates (`golden_ci.json`, e2e) | `ci` (2 seeds, 5–15 epochs) |
| `golden_publication.json` smoke | Publication drift detection | `publication` (2 seeds, 15 epochs) |
| Maintainer workstation | Full publication verdicts | `publication` (10 seeds, 50 epochs) |

Record the exact CPU model and wall-clock time in `results.md` when numbers are cited externally.

---

## Reproducing publication numbers

```bash
source .venv/bin/activate
make health
QML_PROFILE=publication MLFLOW_DISABLE=1 python experiments/exp_021_qml_backend_parity/run.py
make results-new -- --exp exp_021
```

Docker reproduction: `make docker-test` (validates wiring; not full publication profile).

---

## Backend notes

- **`default.qubit`:** always available; backprop gradients.
- **`lightning.qubit`:** requires `pennylane-lightning`; adjoint gradients; may need more RAM on deep circuits.
- Async API jobs (`device: cpu|cuda|auto`) use the same holdout protocol as CLI runs.

See also: [reproducibility.md](reproducibility.md), [zenodo.md](zenodo.md).
