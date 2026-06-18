# Release v0.9.21 — Open Science Release (Phase E)

## Highlights

- **`make open-science-preflight`** — one-command Zenodo/arXiv readiness (release + leaderboard + citation + arXiv sources)
- **Release bundle** now seeds from `publication_experiments.jsonl` (exp_021–024 aligned with paper/leaderboard)
- **`AUTHORS.md`** — author attribution for citation and arXiv
- **Zenodo bundle** includes public leaderboard JSON, QuantumNano-BC model card, exp_024 results

## Phases shipped (30–32 recap)

- Phase 30: exp_024 QuantumNano-BC flagship (30 seeds, clinical baselines)
- Phase 31: publication reproduction CI + paper-artifacts pipeline
- Phase 32: public MicroQML leaderboard on GitHub Pages

## Validation

```bash
make check
make open-science-preflight
make publish-leaderboard-check
```

## Manual follow-ups (author)

1. OSF pre-register exp_024 → paste URL in `hypothesis.md`
2. `git tag v0.9.21 && git push origin v0.9.21` → Zenodo archives release
3. `make finalize-citation DOI=10.5281/zenodo.XXXXXXX`
4. `make arxiv-bundle` → upload to arXiv
5. Optional: HuggingFace model card + weights

See [docs/citation_loop.md](docs/citation_loop.md).
