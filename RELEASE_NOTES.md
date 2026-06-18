# Release v1.2.0 — Large-Scale Open Data & Serve Parity

## Highlights

- **Open datasets** — HIGGS 1.15M (`higgs_v1`) + Synthea CV 1M (`synthea_cv_risk_v1`) with DVC pointers
- **LargeNanoMLP** — `exp_032` ~1.14M params, +14.09 pp val AUC vs logistic on RTX 4060
- **Serve wiring** — batch, REST API, and `score_higgs` chatbot tool share one checkpoint path
- **Parity gate** — `exp_033` max |Δp| 2.98e-07 (batch↔api) on 10K rows
- **Real GPU gate** — `make check-real` (14 tests) on RTX 4060

## Validation (RTX 4060)

```bash
source .venv/bin/activate
make health-gpu
QML_DEVICE=cuda MLFLOW_DISABLE=1 make phase-v1.2.0-preflight
```

## Tag

```bash
git tag v1.2.0
git push origin v1.2.0
```

## Previous releases

- [v1.1.0](docs/releases/v1.1.0.md) — continuous training + chatbot + batch tracks
- [v1.0.0](docs/releases/v1.0.0.md) — real application stack (train + infer)
- [CHANGELOG.md](CHANGELOG.md) — full history from v0.9.x
