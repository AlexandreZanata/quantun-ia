# DVC Remote Setup

Data Version Control (DVC) tracks large artifacts outside Git: experiment exports,
figures, and publication checkpoints. This guide configures a remote for team
replication and Zenodo-adjacent archival.

---

## Prerequisites

```bash
pip install dvc
dvc --version
make health          # confirms .dvc/ is initialized
make dvc-check       # validates dvc.yaml stages and scripts (Phase 27)
```

---

## Option A — Local filesystem remote (solo lab)

Best for single-machine replay before pushing to cloud storage.

```bash
mkdir -p ../quantun-ia-dvc-storage
dvc remote add -d localstore ../quantun-ia-dvc-storage
dvc remote list
```

Push tracked artifacts:

```bash
dvc push
```

Pull on a fresh clone:

```bash
dvc pull
```

Example config: [`.dvc/config.example`](../.dvc/config.example).

---

## Option B — S3-compatible remote

```bash
dvc remote add -d s3store s3://YOUR_BUCKET/quantun-ia
dvc remote modify s3store region us-east-1
# Credentials via AWS CLI, env vars, or IAM role
dvc push
```

---

## Tracked paths (`dvc.yaml`)

| Stage | Output |
|-------|--------|
| `export_results` | `data/exports/results.csv` |
| `figures` | `figures/` |
| `latex_tables` | `paper/tables/` |

Run pipeline after experiments:

```bash
make replay-publication        # full publication_large + exports
# or, from existing logs:
make replay-publication-artifacts
dvc repro                      # equivalent DVC stages for exports only
dvc push
```

---

## Publication replay

One command to regenerate headline publication artifacts:

```bash
make replay-publication-artifacts   # fast — from logs/experiments.jsonl
make replay-publication             # slow — publication_large runs first
```

See [reproducibility.md](reproducibility.md) §4 for runtime estimates.

---

## Security

- Never commit `.dvc/config` with credentials — use env vars or `dvc remote modify --local`.
- `logs/experiments.jsonl` stays gitignored; export golden subset lives in
  `tests/contracts/fixtures/sample_experiments.jsonl` for CI.
