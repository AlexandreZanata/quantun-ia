# Release v0.9.19 — Encoding × Backend Interaction (Phase 28)

## Highlights

- **exp_023** — 2×2 factorial on PCA-MNIST: angle vs amplitude encoding × `default.qubit` vs `lightning.qubit`
- CI smoke + golden bounds guard wiring without publication-profile verdict yet
- `QuantumNetAmplitude` accepts `qml_device` (parity with `QuantumNetBasic`)

## Validation

```bash
make check
MLFLOW_DISABLE=1 pytest tests/integration/test_exp_023_smoke.py -v
```

## Manual follow-ups

- OSF pre-registration before publication-profile runs
- Fill `results.md` after 10-seed publication execution
