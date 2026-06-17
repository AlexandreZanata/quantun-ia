# Citation Loop — Zenodo DOI + arXiv ID

Unified checklist for closing the open-science citation loop after Phases 18b and 19b.
Software version: **v0.9.16** (Phase 25).

---

## Automated preflight

```bash
source .venv/bin/activate
make check
make citation-ready          # version alignment + artifact presence
make release && make release-check   # optional — full Zenodo bundle
make arxiv-bundle-sources    # arXiv TeX tarball (no pdflatex required)
```

`make citation-ready` runs `scripts/validate_citation_ready.py`. Informational messages
for missing `doi:` and `arxiv_id` are expected until manual upload completes.

---

## Step 1 — GitHub release tag

```bash
git tag v0.9.16
git push origin v0.9.16
```

GitHub Actions (`.github/workflows/release.yml`) attaches `dist/release/` artifacts.
See [zenodo.md](zenodo.md) for bundle contents.

---

## Step 2 — Zenodo DOI

1. Enable Zenodo-GitHub integration for this repository.
2. Wait for Zenodo to archive the tagged release.
3. Copy the version DOI (e.g. `10.5281/zenodo.XXXXXXX`).
4. Uncomment and set in `CITATION.cff`:

```yaml
version: 0.9.16
doi: 10.5281/zenodo.XXXXXXX
```

5. Validate:

```bash
pytest tests/contracts/test_citation_cff.py -v
make citation-ready
```

---

## Step 3 — arXiv upload

1. `make paper-build && make arxiv-bundle` (requires TeX Live for PDF).
2. Upload `dist/arxiv/quantun-ia-paper.tar.gz` per [arxiv.md](arxiv.md).
3. Set moderated ID in `paper/arxiv_metadata.yaml`:

```yaml
arxiv_id: "2606.12345"
```

4. Validate:

```bash
pytest tests/contracts/test_arxiv_metadata.py -v
```

---

## Step 4 — Cross-link paper and README

- Add DOI to `paper/references.bib` (`quantunia2026` entry).
- Re-run `make paper-build` if bibliography changed.
- Update root `README.md` citation section with live DOI and arXiv link.

---

## Narrative scope

Primary paper follows **Option C** — see [paper_narrative.md](paper_narrative.md).
Headline experiments: exp_011–014, exp_021–022, negative results 003/009.
Deferred tracks (GV-ALR exp_015, NAS exp_016) are documented separately in
[method_adaptive_lr.md](method_adaptive_lr.md).

---

## Exit evidence (5-star)

| Item | Evidence |
|------|----------|
| Zenodo DOI live | `doi:` uncommented in `CITATION.cff`, `test_citation_cff` green |
| arXiv ID live | `arxiv_id` set in `paper/arxiv_metadata.yaml` |
| Version alignment | `make citation-ready` exit 0 with no blocking errors |
