# Paper draft

LaTeX skeleton for a workshop or journal submission derived from quantun-ia benchmarks.

## Build

```bash
make paper-build   # latex-tables + figures + sync + pdflatex/bibtex
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
| `tables/` | Auto-generated LaTeX from `make latex-tables` |
| `figures/` | Synced from `figures/` via `make paper-sync` |
| `references.bib` | Bibliography |

Methods and limitations are synced from `docs/architecture.md` and the 5-star standard.
Headline narrative: holdout-fair QML benchmark (Option C in `.local/README.md`).

## Submission checklist

- [ ] Run `make paper-build` with publication-profile logs
- [ ] Embed holdout tables from `paper/tables/`
- [ ] Cite Zenodo DOI from `CITATION.cff`
- [ ] Upload to arXiv (cs.LG + quant-ph) or workshop portal
