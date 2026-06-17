# Release v0.9.20 — DVC Remote Bootstrap (Phase 29)

## Highlights

- **`make dvc-setup`** — installs DVC in venv + configures `../quantun-ia-dvc-storage`
- **`make dvc-push`** / **`make dvc-push-full`** — push artifacts without system-wide `dvc` binary
- Fixes `Command 'dvc' not found` when following `docs/dvc_remote.md`

## Validation

```bash
make check
make dvc-setup
make dvc-push
```

## Manual follow-ups

- Zenodo DOI: `make finalize-citation DOI=...`
- arXiv ID after upload
