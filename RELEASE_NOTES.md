# Release v0.9.1 — Zenodo Release Bundle

**Date:** 2026-06-17  
**Codename:** Phase 10 platform layer — Phases 0–10 complete

## Highlights

- **20 experiments** with hypothesis-first workflow (`exp_001`–`exp_020`)
- **REST API** — FastAPI service with SQLite-backed training jobs and multitenancy headers
- **Benchmark PWA** — mobile leaderboard at `/pwa/`
- **Publication pipeline:** figures, LaTeX tables, paper skeleton, Zenodo release bundle with SHA-256 manifest
- **Innovation track:** adaptive LR (015), hybrid NAS (016), poison × topology (017), feature fusion (018)
- **Nano Trainer:** CLI + Streamlit mini training app (019)
- **Engineering:** `make check`, 80% coverage, mypy CI, JSONL contracts, e2e API tests

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
make check
```

## Citation

See [CITATION.cff](CITATION.cff). After Zenodo sync for `v0.9.1`, add the DOI per [docs/zenodo.md](docs/zenodo.md).

## Artifact bundle

```bash
make release        # → dist/release/ with MANIFEST.txt checksums
make release-check  # verify an existing bundle
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
