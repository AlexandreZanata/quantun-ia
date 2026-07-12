# Paper draft

LaTeX skeleton for a workshop or journal submission derived from quantun-ia benchmarks.

## Build

```bash
make paper-build   # latex-tables + figures + sync + pdflatex/bibtex
make arxiv-bundle  # dist/arxiv/quantun-ia-paper.tar.gz for upload
# or step-by-step:
make latex-tables
make figures
make paper-sync
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex
```

## Structure

| Path | Purpose |
|------|---------|
| `main.tex` | Document entry point |
| `sections/` | Introduction, methods, experiments, results, limitations |
| `tables/` | Auto-generated LaTeX from `make latex-tables` + frozen exp_021/022 |
| `figures/` | Synced from `figures/` via `make paper-sync` |
| `references.bib` | Bibliography |
| `arxiv_metadata.yaml` | Title, abstract, categories for arXiv upload |

Methods and limitations are synced from `docs/architecture.md` and the 5-star standard.
Headline narrative: holdout-fair QML benchmark (Option C in `.local/README.md`).

## Submission checklist

- [x] Run `make paper-build` with publication-profile logs (CI `paper-build` job)
- [x] exp_021 + exp_022 `results.md` with uniform statistical sections
- [x] Embed holdout tables (`exp_021_holdout.tex`, `exp_022_holdout.tex` + `make latex-tables`)
- [x] arXiv bundle pipeline (`make arxiv-bundle`, `docs/arxiv.md`)
- [x] Local Cycle v2 arXiv bundle refreshed (F-T4) — includes exp_087 / exp_094 tables
- [ ] Cite Zenodo DOI from `CITATION.cff` after upload (see [docs/zenodo.md](../docs/zenodo.md))
- [ ] Upload to arXiv (cs.LG + quant-ph) — paste `arxiv_id` into `arxiv_metadata.yaml` after moderation

## arXiv ID

<!-- Set after upload: -->
<!-- arxiv:2606.XXXXX -->

See [docs/arxiv.md](../docs/arxiv.md) for the full submission workflow.
