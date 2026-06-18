# Release v1.1.0 — Continuous Training & Application Tracks

## Highlights

- **Continuous retrain** — `exp_027` champion/challenger gate, `make continuous-train`
- **Chatbot integration** — `exp_028` tool adapter vs API (10 dialogues, max |Δp| < 1e-5)
- **Batch pipeline** — `exp_029` batch script vs API on 569-row holdout
- **Scale gate** — `exp_030` 30-seed hybrid stability on circles n=1000
- **Curriculum ablation** — `exp_031` margin curriculum vs random on breast cancer
- **Real GPU gate** — `make check-real` (12 tests) on RTX 4060

## Validation (RTX 4060)

```bash
source .venv/bin/activate
make health-gpu
QML_DEVICE=cuda MLFLOW_DISABLE=1 make check-real
QML_DEVICE=cuda MLFLOW_DISABLE=1 make phase-v1.1.0-preflight
```

## Tag

```bash
git tag v1.1.0
git push origin v1.1.0
```

## Previous releases

- [v1.0.0](docs/releases/v1.0.0.md) — real application stack (train + infer)
- [v1.0.0-rc1](docs/releases/v1.0.0-rc1.md) — open-science preflight (Phase D)
- [CHANGELOG.md](CHANGELOG.md) — full history from v0.9.x
