# Hypothesis — exp_020_api_smoke

## Question

Does the REST API (`POST /api/v1/training-jobs`) produce the same holdout metrics as the direct Nano Trainer path (`train_nanomodel.execute`) for a shared model × dataset pair?

## Expectation vs prior work

- **Different from exp_019:** That experiment validates the Python orchestrator directly. This experiment validates the **HTTP + SQLite persistence** layer on top.
- **Same training core:** Both paths call `train_nanomodel.execute` under the hood.

## Success criteria

- API returns `201` with `status=COMPLETED` for perceptron + breast_cancer + ci.
- Holdout accuracy ∈ [0.35, 1.0].
- `GET /api/v1/training-jobs/{id}` returns the same job with `tenant_id=local`.
- `GET /ready` and `GET /health` return 200.

## Failure criteria

- API unreachable or training job not persisted.
- Accuracy outside sanity bounds.
- Tenant isolation broken (job visible under wrong tenant).
