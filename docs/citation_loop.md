# Citation Loop — Zenodo DOI + arXiv ID

Unified checklist for closing the open-science citation loop.  
Software version: **v0.9.22** (Phase F — Pima generalization release).

---

## Automated preflight

```bash
source .venv/bin/activate
make check
make open-science-preflight   # release + leaderboard + citation + arXiv sources
make citation-ready           # version alignment + artifact presence
```

`make open-science-preflight` runs `scripts/open_science_preflight.py`:

1. Seeds `logs/experiments.jsonl` from `publication_experiments.jsonl` (exp_021–024)
2. Builds `dist/release/` with leaderboard JSON, model card, exp_024 results
3. Verifies public leaderboard (`docs/leaderboard/v1.json`)
4. Validates citation artifacts (`scripts/validate_citation_ready.py`)
5. Bundles arXiv TeX sources (`make arxiv-bundle-sources`)

Informational messages for missing `doi:` and `arxiv_id` are expected until manual upload completes.

---

## Step 1 — GitHub release tag

```bash
git tag v0.9.22
git push origin v0.9.22
```

GitHub Actions (`.github/workflows/release.yml`) attaches `dist/release/` artifacts.  
See [zenodo.md](zenodo.md) for bundle contents.

---

## Step 2 — Zenodo DOI

1. Enable Zenodo-GitHub integration for this repository.
2. Wait for Zenodo to archive tag `v0.9.22`.
3. Copy the version DOI (e.g. `10.5281/zenodo.XXXXXXX`).
4. Apply with one command:

```bash
make finalize-citation DOI=10.5281/zenodo.XXXXXXX
```

5. Validate:

```bash
pytest tests/contracts/test_citation_cff.py tests/contracts/test_arxiv_metadata.py -v
make citation-ready
git commit -am "chore(citation): add Zenodo DOI for v0.9.22"
```

---

## Step 3 — arXiv upload

1. `make paper-build && make arxiv-bundle` (requires TeX Live for PDF).
2. Upload `dist/arxiv/quantun-ia-paper.tar.gz` per [arxiv.md](arxiv.md).
3. Add moderated ID:

```bash
make finalize-citation DOI=10.5281/zenodo.XXXXXXX ARXIV_ID=2606.12345
```

---

## Step 4 — OSF pre-registration (exp_024)

1. File OSF pre-registration for QuantumNano-BC (exp_024).
2. Paste `https://osf.io/...` in `experiments/exp_024_quantum_nano_bc/hypothesis.md`.
3. Add `exp_024_quantum_nano_bc` to `PREREG_REQUIRED` in `tests/contracts/test_osf_prereg.py`.

---

## Exit evidence (5-star)

| Item | Evidence |
|------|----------|
| Zenodo DOI live | `doi:` in `CITATION.cff`, `test_citation_cff` green |
| arXiv ID live | `arxiv_id` in `paper/arxiv_metadata.yaml` |
| Release bundle | `make open-science-preflight` exit 0 |
| Public leaderboard | https://alexandrezanata.github.io/quantun-ia/leaderboard/ |
| Version alignment | `make citation-ready` exit 0 (no blocking errors) |
