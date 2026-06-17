# arXiv Submission Guide

This guide describes how to submit the quantun-ia paper draft to [arXiv](https://arxiv.org)
(cs.LG + quant-ph). Software version **v0.9.13** (Phase 19b).

---

## Prerequisites

- arXiv account with endorsement for `cs.LG` or `quant-ph`
- Local `make check` green
- `make paper-build` produces `paper/main.pdf`
- Metadata in `paper/arxiv_metadata.yaml` reviewed

---

## Step 1 — Build paper and bundle

```bash
source .venv/bin/activate
make check
make paper-build          # latex-tables + figures + PDF
make arxiv-bundle         # dist/arxiv/quantun-ia-paper.tar.gz
```

The bundle includes:

- `main.tex`, `sections/`, `tables/`, `figures/`, `references.bib`
- `00README.txt` with compile instructions
- `main.pdf` (reference copy)
- `arxiv_metadata.yaml`

---

## Step 2 — Upload to arXiv

1. Go to [arxiv.org/submit](https://arxiv.org/submit).
2. Upload `dist/arxiv/quantun-ia-paper.tar.gz` or individual TeX sources.
3. Copy **title**, **abstract**, **authors**, and **categories** from `paper/arxiv_metadata.yaml`.
4. Set comments field from metadata `comments` block.
5. Select license (recommended: arXiv non-exclusive distribution).

---

## Step 3 — Record arXiv ID

After moderation, set the ID in `paper/arxiv_metadata.yaml`:

```yaml
arxiv_id: "2606.12345"
```

Update `paper/README.md` submission checklist and commit:

```bash
pytest tests/contracts/test_arxiv_metadata.py -v
```

---

## Step 4 — Cross-link Zenodo

When the Zenodo DOI is live (Phase 18b), cite it in `references.bib` and re-run `make paper-build`.

---

## CI

The `paper-build` GitHub Actions job runs `make paper-build` and `make arxiv-bundle --skip-pdf` on fixture logs.

---

## Notes

- Holdout tables for exp\_021/022 are publication-frozen in `paper/tables/exp_021_holdout.tex` and `exp_022_holdout.tex`.
- Other experiment tables are regenerated from `logs/experiments.jsonl` via `make latex-tables`.
- arXiv does not accept GitHub URLs as data — cite the Zenodo release for artifact hashes.
