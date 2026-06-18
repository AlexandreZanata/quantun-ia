# Release v1.0.0-rc1 — Real Application + Open Science (Phases B–D)

## Highlights

- **Real GPU gate** — `make check-real` on RTX 4060 (6 tests: nanotrainer, exp_026 API=CLI, publication bounds)
- **exp_026** — async API (`device=cuda`) matches CLI holdout (0.00 pp max delta, 5 seeds)
- **Phase C publication refresh** — exp_024/025 re-run on RTX 4060 (30 seeds, ~2 min combined)
- **Open-science bundle** — `make phase-d-preflight` builds Zenodo-ready `dist/release/` + arXiv sources
- **QuantumNano-BC** — hybrid 97.4% vs logistic 97.9% (breast cancer); Pima 76.2% vs 77.2%

## Validation (RTX 4060)

```bash
source .venv/bin/activate
make health-gpu
QML_DEVICE=cuda MLFLOW_DISABLE=1 make phase-d-preflight
```

## Manual steps (after tag push)

1. `git tag v1.0.0-rc1 && git push origin v1.0.0-rc1`
2. Zenodo archives the tag → `make finalize-citation DOI=10.5281/zenodo.XXXXXXX`
3. arXiv upload → `make finalize-citation DOI=... ARXIV_ID=2606.XXXXX`

## Previous releases

See [CHANGELOG.md](CHANGELOG.md) — v0.9.22 (Phase F Pima), v0.9.21 (open-science preflight).
