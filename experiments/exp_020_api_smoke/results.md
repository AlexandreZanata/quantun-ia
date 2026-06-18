# Results — EXP 020 (API Smoke)

**Run date:** 2026-06-18  
**Profile:** `ci` (infrastructure validation — not a publication benchmark)  
**Pair:** perceptron + breast_cancer via `POST /api/v1/training-jobs`

## Holdout results

| Endpoint | Status | Holdout accuracy |
|----------|--------|------------------|
| `POST /api/v1/training-jobs` | `201 COMPLETED` | 84.2% |
| `GET /api/v1/training-jobs/{id}` | `200` | job persisted |
| `GET /health` | `200` | liveness OK |
| `GET /ready` | `200` | readiness OK |

Tenant header: `X-Tenant-ID: local`. SQLite database: `data/exp_020_api.db`.

## Verdict

**accepted** — REST API training job completes with holdout accuracy within [35%, 100%] and persists retrievable job state.

## Conclusion

Infrastructure smoke validates the HTTP + SQLite layer on top of `train_nanomodel.execute` (Phase 10). Training core matches exp_019; this experiment adds API contract and persistence checks only.

## Limitations

- Synchronous training in TestClient — production async queue not stress-tested here.
- Single model × dataset pair (perceptron + breast_cancer).
- JWT auth not exercised in this smoke (see `tests/e2e/test_api_auth_async.py`).
