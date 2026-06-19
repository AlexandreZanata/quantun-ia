# GoBug File-Level Defect Dataset

**Purpose:** Software defect prediction on Go file-level static metrics (Phase 2 / exp_045).

## Source

- **GoBug:** [IEEE DataPort](https://doi.org/10.21227/bk5q-fs89)
- **Repro:** [ecylmz/go-bug-collector](https://github.com/ecylmz/go-bug-collector) `file_data/combined/`
- **Builder:** `scripts/build_gobug_subset.py`

## Cohort (v1)

| Field | Value |
|-------|-------|
| Rows | ~38,818 (combined release) |
| Features | 23 Go/static metrics |
| Label | file-level defect (`1`) vs clean (`0`) |
| Prevalence | ~31% positive |
| Split | Temporal proxy — sort by `sha`, 70/15/15 contiguous |

## Build

```bash
source .local/env.sh
make data-open-gobug
make data-open-verify
```

## Citation

GoBug dataset via go-bug-collector. See `LICENSE-data` in upstream repo.
