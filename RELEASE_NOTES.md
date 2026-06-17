# Release v0.9.18 — DVC Artifact Validation (Phase 27)

**Date:** 2026-06-17

## Highlights

- **`make dvc-check`** — validates `dvc.yaml` stages align with release pipeline
- Remote setup remains manual per `docs/dvc_remote.md`
- **290+ tests** green via `make check`

## Preflight

```bash
make dvc-check
make health
```

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
