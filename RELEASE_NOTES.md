# Release v1.0.0 — QuantumNano-BC Real Application

## Highlights

- **Real GPU gate** — `make check-real` on RTX 4060 (7 tests)
- **Train + infer** — `hybrid_sandwich` checkpoint, scaler, `POST /api/v1/predictions`
- **CLI demo** — `make train-ship` trains on full breast cancer and scores sample rows
- **Scientific parity** — exp_024/025/026 validated on real clinical tabular data
- **Open science** — `make phase-d-preflight` builds release bundle + arXiv sources

## Validation (RTX 4060)

```bash
source .venv/bin/activate
make health-gpu
QML_DEVICE=cuda MLFLOW_DISABLE=1 make check-real
QML_DEVICE=cuda MLFLOW_DISABLE=1 make train-ship
```

## Tag

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Previous releases

- [v1.0.0-rc1](docs/releases/v1.0.0-rc1.md) — open-science preflight (Phase D)
- [CHANGELOG.md](CHANGELOG.md) — full history from v0.9.x
