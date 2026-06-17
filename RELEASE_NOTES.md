# Release v0.4.0 — 5-Star Lab Closure

**Date:** 2026-06-17  
**Codename:** Full lab maturity — Phases 0–5 complete

## Highlights

- **15 experiments** with hypothesis-first workflow (`exp_001`–`exp_015`)
- **Publication pipeline:** figures, LaTeX tables, paper skeleton, Zenodo guide
- **Innovation:** exp_015 gradient-variance adaptive LR + literature review
- **Engineering:** `make check`, 80% coverage, mypy CI, JSONL contracts, `qml-run` CLI
- **Real benchmarks:** UCI breast cancer, MNIST PCA, sequence baselines
- **results.md** for exp_011–exp_015 (publication profile)

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
make check
```

## Citation

See [CITATION.cff](CITATION.cff). After Zenodo sync, add DOI per [docs/zenodo.md](docs/zenodo.md).

## Artifact bundle

```bash
make release   # → dist/release/
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
