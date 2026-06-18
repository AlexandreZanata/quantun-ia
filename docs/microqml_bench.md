# MicroQML Bench v1

**MicroQML Bench** is the versioned, citable benchmark suite for holdout-fair quantum vs classical comparisons in the quantun-ia lab.

## Version

| Field | Value |
|-------|-------|
| Bench ID | `microqml_bench` |
| Version | **1.0.0** |
| Schema | **v1** |
| Config | `config/microqml_bench/v1.yaml` |

## Task categories (v1)

| Task ID | Category | Experiment | Dataset |
|---------|----------|------------|---------|
| `synthetic_circles` | synthetic | exp_001 | circles |
| `tabular_breast_cancer` | tabular | exp_011 | breast cancer |
| `quantum_nano_bc` | tabular | exp_024 | breast cancer (flagship) |
| `pca_mnist_binary` | image_tabular | exp_012 | MNIST 0 vs 1 (PCA-8) |
| `sequence_phase` | sequence | exp_014 | sequential_phase |

All tasks use the **publication profile**: 30% stratified holdout (fit preprocessing on train only), bootstrap 95% CI, Holm-corrected Wilcoxon for paired comparisons. Flagship task `quantum_nano_bc` uses **30 seeds** (exp_024).

## Public leaderboard (GitHub Pages)

| Resource | URL |
|----------|-----|
| Viewer | https://alexandrezanata.github.io/quantun-ia/leaderboard/ |
| JSON | https://alexandrezanata.github.io/quantun-ia/leaderboard/v1.json |
| Source | `docs/leaderboard/` (committed, deployed via `.github/workflows/pages.yml`) |

```bash
make publish-leaderboard        # regenerate from publication fixture
make publish-leaderboard-check  # CI validation
```

External consumers should validate against `tests/contracts/microqml_bench_schema.py`.

## Export bundle

```bash
make microqml-bench
# → dist/microqml_bench/v1.json
```

Or via REST API:

```bash
curl http://127.0.0.1:8000/api/v1/benchmarks/microqml/v1
```

## JSON schema

Contract: `tests/contracts/microqml_bench_schema.py`  
Fixture: `tests/contracts/fixtures/sample_microqml_bench_v1.json`  
CI validates every export with `jsonschema`.

Required top-level fields: `bench_id`, `version`, `schema_version`, `protocol`, `tasks`, `leaderboard`, `generated_at`.

## Leaderboard rows

Each row maps one model on one bench task:

- `accuracy_pct` — holdout accuracy (percent)
- `ci_low_pct` / `ci_high_pct` — bootstrap 95% CI when multi-seed summary exists
- `source` — `multi_seed_summary` or `single_seed`
- `eval_set` — always `holdout_test` for publication rows

Rows are filtered to the five v1 primary tasks; `nano_train` and smoke experiments are excluded.

## Reproduction

External replicators should:

1. Clone quantun-ia and install: `make install`
2. Run headline experiments: `make experiments-new` (or individual exp_001/011/012/014)
3. Export bench: `make microqml-bench`
4. Validate: `pytest tests/contracts/test_microqml_bench_contract.py -v`

Target runtime on CPU: under 1 hour for CI smoke subset; full publication profile may take longer (document hardware in `results.md`).

## Citation

When citing MicroQML Bench results, include:

- Bench version (`1.0.0`) and schema version (`1`)
- Software version from the export (`software_version` field)
- [CITATION.cff](../CITATION.cff) for the quantun-ia codebase

## Hugging Face / Zenodo

- **Zenodo:** bundle `dist/microqml_bench/v1.json` with release artifacts (`make release`)
- **Hugging Face dataset card:** use the task table above as metadata; synthetic task generators live in `src/data/generators.py`

## Versioning policy

- **Patch** (1.0.x): schema unchanged; leaderboard data refresh only
- **Minor** (1.x.0): new tasks added with backward-compatible schema v1
- **Major** (2.0.0): new `schema_version` with breaking field changes
