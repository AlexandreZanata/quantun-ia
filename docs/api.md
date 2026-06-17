# REST API

Phase 10 adds a FastAPI service wrapping the Nano Trainer orchestrator with SQLite job persistence and a mobile-friendly benchmark PWA.

## Quick start

```bash
make install
make api
# → http://127.0.0.1:8000/health
# → http://127.0.0.1:8000/pwa/  (mobile leaderboard)
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (always 200) |
| GET | `/ready` | Readiness — DB + nanotrainer config |
| GET | `/metrics` | Prometheus-style counters |
| POST | `/api/v1/training-jobs` | Run mini training, persist job |
| GET | `/api/v1/training-jobs/{id}` | Fetch job by id (tenant-scoped) |
| GET | `/api/v1/benchmarks/leaderboard` | Publication leaderboard JSON |
| GET | `/pwa/` | Static PWA benchmark viewer |

## Training job request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/training-jobs \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: local" \
  -d '{"model_name":"perceptron","dataset":"breast_cancer","profile":"ci","epochs":5}'
```

Response `201` includes `status` (`COMPLETED` or `FAILED`), `result`, and `id` for follow-up GET.

## Multitenancy

- Send `X-Tenant-ID` on every authenticated-style request (default: `local`).
- All repository queries filter by `tenant_id`.
- Jobs are soft-deletable via `deleted_at` (no hard deletes).

## Persistence

SQLite schema: `src/infrastructure/database/schema.sql`  
Default path: `data/quantun-ia.db` (override with `DATABASE_PATH`).

## Architecture

```
HTTP (FastAPI)
  → src/application/create_training_job.py
      → SqliteTrainingJobRepository
      → train_nanomodel.execute()
  → logs/experiments.jsonl
```

## Validation

- `experiments/exp_020_api_smoke/` — API path smoke test
- `tests/e2e/test_api_routes.py` — route contract tests
- `make api-demo` — run exp_020 locally

## Limitations

- Synchronous training in request thread (use `mini`/`ci` profiles).
- No JWT auth yet — header-based `tenantId` only.
- TypeScript/Fastify backend remains future work (Phase 10+ backlog).
