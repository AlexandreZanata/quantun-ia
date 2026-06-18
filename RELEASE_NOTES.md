# Release v0.9.22 — Pima Generalization (Phase F)

## Highlights

- **exp_025 Pima Indians Diabetes** — 30-seed publication parity vs logistic regression (Δ=−1.0 pp)
- **`pima_diabetes` dataset** — OpenML id=37 loader, MicroQML Bench task, Zenodo CSV export
- **RTX 4060 local profile** — `QML_DEVICE=cuda` for classical models; PennyLane hybrids stay CPU-safe
- **`tests/conftest.py`** — auto-prefers CUDA when NVIDIA GPU is available

## Validation

```bash
source .venv/bin/activate
QML_DEVICE=cuda MLFLOW_DISABLE=1 make check
QML_DEVICE=cuda MLFLOW_DISABLE=1 python experiments/exp_025_pima_generalization/run.py --profile ci
```

## Previous release

See [v0.9.21](CHANGELOG.md#0921---2026-06-18) — open-science preflight (Phase E).
