# Paper draft

LaTeX skeleton for a workshop or journal submission derived from quantun-ia benchmarks.

## Build

```bash
# Generate tables from experiment logs first
make latex-tables
make figures
cp -r figures paper/figures   # optional: embed PDFs

cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Structure

| Path | Purpose |
|------|---------|
| `main.tex` | Document entry point |
| `sections/` | Introduction, methods, experiments, results |
| `tables/` | Auto-generated LaTeX from `make latex-tables` |
| `figures/` | Copy from `figures/` after `make figures` |
| `references.bib` | Bibliography |

Methods content is manually synced from `docs/architecture.md` (automate in a future phase).
