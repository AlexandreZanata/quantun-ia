# Release v0.9.17 — Citation Finalizer (Phase 26)

**Date:** 2026-06-17

## Highlights

- **`make finalize-citation`** — paste Zenodo DOI (+ optional arXiv ID) in one command
- **Release record** — `docs/releases/v0.9.16.md` documents pushed tag evidence
- Tag `v0.9.16` already on `origin`; DOI paste is the remaining manual step

## After Zenodo sync

```bash
make finalize-citation DOI=10.5281/zenodo.XXXXXXX ARXIV_ID=2606.XXXXX
make citation-ready
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
