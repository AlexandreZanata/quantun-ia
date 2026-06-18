# Zenodo Release Guide

This guide describes how to obtain a citable DOI for quantun-ia releases via the
[Zenodo–GitHub integration](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).

**Current target version:** `v0.9.21` (Phases 30–32 + open science release).

---

## Prerequisites

- Admin access to the GitHub repository
- Zenodo account linked to GitHub
- Local `make check` green
- `make open-science-preflight` green
- Publication `results.md` for headline experiments (exp_021–024 included in bundle)

---

## Step 1 — Prepare release artifacts

```bash
source .venv/bin/activate
make health
make check
make open-science-preflight
make release-check      # verify SHA-256 checksums in MANIFEST.txt
```

This creates `dist/release/` containing:

- `results.csv` — aggregated experiment export
- `reference_datasets/` — `breast_cancer.csv`, `circles.csv` + metadata JSON
- `figures/*.pdf` — publication figures
- `tables/*.tex` — LaTeX holdout summary tables
- `microqml_bench/v1.json` — MicroQML Bench export (from publication fixture)
- `leaderboard/v1.json` — public GitHub Pages leaderboard snapshot
- `model_cards/quantum_nano_bc.md` — flagship model card
- `experiments/exp_021_qml_backend_parity/results.md` through `exp_024_quantum_nano_bc/results.md`
- `docs/api.md`, `compute_environment.md`, `ethics.md`, `microqml_bench.md`, `zenodo.md`
- `publication_large_summary.md` (if present)
- `requirements.lock`
- `AUTHORS.md`, `CITATION.cff`, `RELEASE_NOTES.md`, `CHANGELOG.md`, `SECURITY.md`
- `MANIFEST.txt` — relative paths with SHA-256 checksums

---

## Step 2 — Enable Zenodo on GitHub

1. Log in to [zenodo.org](https://zenodo.org) with your GitHub account.
2. Go to **Account → GitHub** and enable the `quantun-ia` repository.
3. Zenodo will archive each GitHub release automatically.

---

## Step 3 — Create GitHub release v0.9.12

```bash
git tag v0.9.12
git push origin v0.9.12
```

GitHub Actions (`.github/workflows/release.yml`) will:

1. Build `dist/release/` with manifest verification
2. Attach artifacts to the GitHub release using `RELEASE_NOTES.md`

Zenodo will create an archive within a few minutes.

---

## Step 4 — Update CITATION.cff

Copy the DOI from the Zenodo record page and add to `CITATION.cff`:

```yaml
version: 0.9.12
date-released: 2026-06-17
doi: 10.5281/zenodo.XXXXXXX
```

Commit and push the DOI update. Contract tests validate the DOI format when present.

```bash
pytest tests/contracts/test_citation_cff.py -v
```

---

## Step 5 — Verify

- [ ] Zenodo record shows correct version and files
- [ ] `CITATION.cff` includes live DOI (test no longer skipped)
- [ ] README citation section references DOI
- [ ] Paper draft (`paper/main.tex`) cites the Zenodo archive
- [ ] `make release-check` passes on the published bundle

---

## Notes

- Zenodo archives the **full repository snapshot** at release time plus uploaded assets.
- The DOI is version-specific; each new release gets a new DOI (with a concept DOI for all versions).
- Weekly CI cron runs integration smoke on Mondays to catch dependency drift.
