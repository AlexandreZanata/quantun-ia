# Zenodo Release Guide

This guide describes how to obtain a citable DOI for quantun-ia releases via the
[Zenodo–GitHub integration](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content).

---

## Prerequisites

- Admin access to the GitHub repository
- Zenodo account linked to GitHub
- Local experiment logs (`logs/experiments.jsonl`) or exported CSV from DVC

---

## Step 1 — Prepare release artifacts

```bash
source .venv/bin/activate
make release
```

This creates `dist/release/` containing:

- `results.csv` — aggregated experiment export
- `figures/*.pdf` — publication figures
- `tables/*.tex` — LaTeX holdout summary tables
- `publication_large_summary.md`
- `requirements.lock`
- `MANIFEST.txt`

---

## Step 2 — Enable Zenodo on GitHub

1. Log in to [zenodo.org](https://zenodo.org) with your GitHub account.
2. Go to **Account → GitHub** and enable the `quantun-ia` repository.
3. Zenodo will archive each GitHub release automatically.

---

## Step 3 — Create GitHub release v0.2.0

```bash
git tag v0.2.0
git push origin v0.2.0
```

On GitHub:

1. **Releases → Draft a new release**
2. Tag: `v0.2.0`
3. Title: `v0.2.0 — Publication artifacts (Phase 3)`
4. Attach files from `dist/release/`
5. Publish release

Zenodo will create an archive within a few minutes.

---

## Step 4 — Update CITATION.cff

Copy the DOI from the Zenodo record page and add to `CITATION.cff`:

```yaml
version: 0.2.0
date-released: 2026-06-17
doi: 10.5281/zenodo.XXXXXXX
```

Commit and push the DOI update.

---

## Step 5 — Verify

- [ ] Zenodo record shows correct version and files
- [ ] `CITATION.cff` includes DOI
- [ ] README citation section references DOI
- [ ] Paper draft (`paper/main.tex`) cites the Zenodo archive

---

## Notes

- Zenodo archives the **full repository snapshot** at release time plus uploaded assets.
- The DOI is version-specific; each new release gets a new DOI (with a concept DOI for all versions).
- If logs are empty on the release machine, figures will be empty — run `make experiment-large` first or use `dvc pull`.
