# REST API

Phase 10 adds a FastAPI service wrapping the Nano Trainer orchestrator with SQLite job persistence and a mobile-friendly benchmark PWA.

Phase 16 adds **JWT RS256 auth**, **refresh token rotation**, and an **async job worker** with optional GPU device selection.

## Quick start

```bash
make install
make api
# → http://127.0.0.1:8000/health
# → http://127.0.0.1:8000/pwa/  (mobile leaderboard)
```

## Authentication (Phase 16)

When `API_AUTH_REQUIRED=1`, protected routes require a Bearer access token. Issue tokens with your tenant API key:

```bash
export API_AUTH_SECRET=dev-secret-change-me   # override in production

curl -X POST http://127.0.0.1:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"local","api_key":"dev-secret-change-me"}'
```

Use the returned `access_token`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/training-jobs \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"perceptron","dataset":"breast_cancer","profile":"ci","epochs":5}'
```

Refresh rotation:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"<refresh_token>"}'
```

| Setting | Default | Purpose |
|---------|---------|---------|
| `API_AUTH_REQUIRED` | `0` | Require JWT on training job routes |
| `API_AUTH_SECRET` | `dev-secret-change-me` | API key for `/auth/token` |
| `JWT_PRIVATE_KEY_PEM` | auto-generated dev key | RS256 signing key |
| `JWT_PUBLIC_KEY_PEM` | paired public key | RS256 verification key |

When `API_AUTH_REQUIRED=0` (default), `X-Tenant-ID` header fallback remains for local smoke tests.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (always 200) |
| GET | `/ready` | Readiness — DB + nanotrainer config |
| GET | `/metrics` | Prometheus-style counters |
| POST | `/api/v1/auth/token` | Issue access + refresh tokens |
| POST | `/api/v1/auth/refresh` | Rotate refresh token |
| POST | `/api/v1/training-jobs` | Create job (sync 201 or async 202) |
| GET | `/api/v1/training-jobs/{id}` | Fetch job by id (tenant-scoped) |
| GET | `/api/v1/benchmarks/leaderboard` | Publication leaderboard JSON |
| GET | `/pwa/` | Static PWA benchmark viewer |

## Training job request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/training-jobs \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: local" \
  -d '{"model_name":"perceptron","dataset":"breast_cancer","profile":"ci","epochs":5,"device":"cpu"}'
```

Async mode (returns `202 PENDING`; background worker completes the job):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/training-jobs \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: local" \
  -d '{"model_name":"perceptron","dataset":"breast_cancer","profile":"ci","async_mode":true,"device":"cpu"}'
```

Poll `GET /api/v1/training-jobs/{id}` until `status` is `COMPLETED` or `FAILED`.

Response `201` (sync) or `202` (async) includes `status`, `device`, `result` (when complete), and `id`.

## Multitenancy

- JWT access tokens carry `tenantId`; legacy mode uses `X-Tenant-ID` (default: `local`).
- All repository queries filter by `tenant_id`.
- Jobs are soft-deletable via `deleted_at` (no hard deletes).

## Persistence

SQLite schema: `src/infrastructure/database/schema.sql`  
Tables: `training_jobs`, `refresh_tokens`  
Default path: `data/quantun-ia.db` (override with `DATABASE_PATH`).

## Architecture

```
HTTP (FastAPI)
  → auth/token + auth/refresh (JWT RS256)
  → create_training_job (sync or enqueue)
      → TrainingJobWorker (async PENDING → RUNNING → COMPLETED)
      → process_training_job → train_nanomodel.execute()
  → SqliteTrainingJobRepository (tenant-scoped)
  → logs/experiments.jsonl
```

## Validation

- `experiments/exp_020_api_smoke/` — API path smoke test
- `tests/e2e/test_api_routes.py` — legacy header auth routes
- `tests/e2e/test_api_auth_async.py` — JWT + async queue (Phase 16)
- `make api-demo` — run exp_020 locally

## Limitations

- Async worker is in-process (SQLite poll); production scale-out needs Redis/PostgreSQL (Phase 17+).
- Refresh tokens stored in SQLite (not httpOnly cookies yet).
- TypeScript/Fastify backend remains future work for multitenancy at scale.
