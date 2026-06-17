# Release v0.9.13 — arXiv Submission Pipeline (Phases 0–22b + 19b)

**Date:** 2026-06-17  
**Codename:** 5-star lab — paper bundle ready for arXiv upload

## Highlights

- **arXiv pipeline** — `make arxiv-bundle`, `paper/arxiv_metadata.yaml`, `docs/arxiv.md`
- **Paper draft** — exp_021/022 frozen holdout tables, embedded figures, CI `paper-build`
- **22 experiments** with hypothesis-first workflow and uniform `results.md`
- **MicroQML Bench v1** — versioned JSON schema, `make microqml-bench`
- **Nano Parity Bench** — exp_022 publication (honest inconclusive verdict)
- **exp_021** — PennyLane backend parity accepted (Δ=−0.4 pp, 10 seeds)
- **Engineering:** 245+ tests, 80% coverage, e2e in CI, weekly smoke cron

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
make check
```

## Citation

See [CITATION.cff](CITATION.cff). After Zenodo sync:

1. Copy DOI from [zenodo.org](https://zenodo.org) → uncomment `doi:` in `CITATION.cff`
2. Run `pytest tests/contracts/test_citation_cff.py -v`

Guides: [docs/zenodo.md](docs/zenodo.md), [docs/arxiv.md](docs/arxiv.md).

## Artifact bundles

```bash
make release        # → dist/release/ with MANIFEST.txt checksums
make arxiv-bundle   # → dist/arxiv/quantun-ia-paper.tar.gz
git tag v0.9.13 && git push origin v0.9.13   # triggers .github/workflows/release.yml
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
