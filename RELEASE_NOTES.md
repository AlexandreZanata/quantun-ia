# Release v0.9.12 — Zenodo Citation Loop (Phases 0–22b)

**Date:** 2026-06-17  
**Codename:** 5-star lab — publication verdicts, MicroQML Bench, Nano Parity Bench

## Highlights

- **22 experiments** with hypothesis-first workflow and uniform `results.md` (exp_011–018, exp_021, exp_022)
- **MicroQML Bench v1** — versioned JSON schema, `make microqml-bench`, REST export
- **Nano Parity Bench** — `qml-bench-parity`, exp_022 publication (honest inconclusive verdict)
- **exp_021** — PennyLane backend parity accepted (Δ=−0.4 pp, 10 seeds)
- **Platform:** JWT auth, async GPU job queue, e2e in CI, benchmark PWA
- **Compliance:** ORCID, OSF prereg contracts, `docs/ethics.md`, `docs/compute_environment.md`
- **Engineering:** 230+ tests, 80% coverage, `golden_publication.json`, weekly smoke cron

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
make check
```

## Citation

See [CITATION.cff](CITATION.cff). After Zenodo sync for `v0.9.12`:

1. Copy DOI from [zenodo.org](https://zenodo.org) → uncomment `doi:` in `CITATION.cff`
2. Run `pytest tests/contracts/test_citation_cff.py -v`

Full guide: [docs/zenodo.md](docs/zenodo.md).

## Artifact bundle

```bash
make release        # → dist/release/ with MANIFEST.txt checksums
make release-check  # verify an existing bundle
git tag v0.9.12 && git push origin v0.9.12   # triggers .github/workflows/release.yml
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
