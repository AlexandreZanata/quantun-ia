"""FastAPI application — Nano Trainer REST API and benchmark PWA."""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from dashboard.benchmark_data import load_records, to_leaderboard_rows
from src.application.create_training_job import CreateTrainingJobDTO
from src.application.create_training_job import execute as create_job
from src.application.issue_tokens import IssueTokensDTO
from src.application.issue_tokens import execute as issue_tokens
from src.application.nanotrainer_config import load_nanotrainer_config
from src.application.refresh_tokens import RefreshTokensDTO
from src.application.refresh_tokens import execute as refresh_tokens_execute
from src.benchmark.microqml_bench import build_bench_export
from src.infrastructure.database.connection import connect, init_schema
from src.infrastructure.database.repositories.sqlite_refresh_token_repository import (
    SqliteRefreshTokenRepository,
)
from src.infrastructure.database.repositories.sqlite_training_job_repository import (
    SqliteTrainingJobRepository,
)
from src.infrastructure.queue.training_job_worker import TrainingJobWorker
from src.presentation.http.dependencies import TenantContext, resolve_tenant_context
from src.presentation.http.schemas import (
    CreateTrainingJobRequest,
    IssueTokenRequest,
    LeaderboardResponse,
    LeaderboardRowResponse,
    RefreshTokenRequest,
    TokenPairResponse,
    TrainingJobResponse,
)
from src.shared.result import Fail, Ok

PWA_DIR = Path(__file__).resolve().parents[3] / "dashboard" / "static" / "pwa"
_REQUEST_COUNT = 0
_ERROR_COUNT = 0
_ROUTE_LATENCY: dict[str, list[float]] = {}


def _db_path() -> Path:
    return Path(os.environ.get("DATABASE_PATH", "data/quantun-ia.db"))


def _job_to_response(job: Any) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=job.id,
        tenant_id=job.tenant_id,
        model_name=job.model_name,
        dataset=job.dataset,
        profile=job.profile,
        status=job.status.value,
        exp_id=job.exp_id,
        seed=job.seed,
        epochs=job.epochs,
        device=job.device,
        result=job.result,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        version=job.version,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = connect(_db_path())
    init_schema(conn)
    app.state.db_conn = conn
    worker = TrainingJobWorker(_db_path())
    worker.start()
    app.state.job_worker = worker
    yield
    worker.stop()
    conn.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="quantun-ia API",
        version="0.9.10",
        description="REST API for Nano Trainer and benchmark viewing",
        lifespan=lifespan,
    )

    if PWA_DIR.exists():
        app.mount("/pwa", StaticFiles(directory=str(PWA_DIR), html=True), name="pwa")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Callable):
        global _REQUEST_COUNT, _ERROR_COUNT
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        route = request.url.path
        _REQUEST_COUNT += 1
        _ROUTE_LATENCY.setdefault(route, []).append(elapsed)
        if response.status_code >= 400:
            _ERROR_COUNT += 1
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> dict[str, str]:
        conn: sqlite3.Connection = request.app.state.db_conn
        conn.execute("SELECT 1")
        load_nanotrainer_config()
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        lines = [
            "# HELP http_requests_total Total HTTP requests",
            "# TYPE http_requests_total counter",
            f"http_requests_total {_REQUEST_COUNT}",
            "# HELP http_errors_total Total HTTP 4xx/5xx responses",
            "# TYPE http_errors_total counter",
            f"http_errors_total {_ERROR_COUNT}",
        ]
        for route, samples in _ROUTE_LATENCY.items():
            if not samples:
                continue
            avg = sum(samples) / len(samples)
            safe_route = route.replace('"', "")
            lines.append(f'http_route_latency_seconds{{route="{safe_route}"}} {avg:.6f}')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/api/v1/auth/token", response_model=TokenPairResponse)
    def post_auth_token(body: IssueTokenRequest, request: Request) -> TokenPairResponse:
        conn: sqlite3.Connection = request.app.state.db_conn
        refresh_repo = SqliteRefreshTokenRepository(conn)
        outcome = issue_tokens(
            IssueTokensDTO(tenant_id=body.tenant_id, api_key=body.api_key, user_id=body.user_id),
            refresh_repo,
        )
        if isinstance(outcome, Fail):
            raise HTTPException(
                status_code=401,
                detail={"code": outcome.error.code, "message": outcome.error.message},
            )
        assert isinstance(outcome, Ok)
        pair = outcome.value
        return TokenPairResponse(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )

    @app.post("/api/v1/auth/refresh", response_model=TokenPairResponse)
    def post_auth_refresh(body: RefreshTokenRequest, request: Request) -> TokenPairResponse:
        conn: sqlite3.Connection = request.app.state.db_conn
        refresh_repo = SqliteRefreshTokenRepository(conn)
        outcome = refresh_tokens_execute(RefreshTokensDTO(refresh_token=body.refresh_token), refresh_repo)
        if isinstance(outcome, Fail):
            raise HTTPException(
                status_code=401,
                detail={"code": outcome.error.code, "message": outcome.error.message},
            )
        assert isinstance(outcome, Ok)
        pair = outcome.value
        return TokenPairResponse(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )

    @app.post(
        "/api/v1/training-jobs",
        response_model=TrainingJobResponse,
        status_code=201,
        responses={202: {"model": TrainingJobResponse}},
    )
    def post_training_job(
        body: CreateTrainingJobRequest,
        request: Request,
        response: Response,
        tenant: TenantContext = Depends(resolve_tenant_context),
    ) -> TrainingJobResponse:
        conn: sqlite3.Connection = request.app.state.db_conn
        repo = SqliteTrainingJobRepository(conn)
        dto = CreateTrainingJobDTO(
            tenant_id=tenant.tenant_id,
            model_name=body.model_name,
            dataset=body.dataset,
            profile=body.profile,
            epochs=body.epochs,
            seed=body.seed,
            exp_id=body.exp_id,
            device=body.device,
            async_mode=body.async_mode,
        )
        os.environ.setdefault("MLFLOW_DISABLE", "1")
        outcome = create_job(dto, repo)
        if isinstance(outcome, Fail):
            raise HTTPException(
                status_code=400,
                detail={"code": outcome.error.code, "message": outcome.error.message},
            )
        assert isinstance(outcome, Ok)
        job = outcome.value
        if body.async_mode:
            response.status_code = 202
        return _job_to_response(job)

    @app.get("/api/v1/training-jobs/{job_id}", response_model=TrainingJobResponse)
    def get_training_job(
        job_id: str,
        request: Request,
        tenant: TenantContext = Depends(resolve_tenant_context),
    ) -> TrainingJobResponse:
        conn: sqlite3.Connection = request.app.state.db_conn
        repo = SqliteTrainingJobRepository(conn)
        job = repo.find_by_id(job_id, tenant.tenant_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return _job_to_response(job)

    @app.get("/api/v1/benchmarks/leaderboard", response_model=LeaderboardResponse)
    def get_leaderboard() -> LeaderboardResponse:
        records = load_records()
        rows = to_leaderboard_rows(records)
        return LeaderboardResponse(
            rows=[
                LeaderboardRowResponse(
                    exp_id=r["exp_id"],
                    model=r["model"],
                    accuracy=r["accuracy"],
                    ci_low=r.get("ci_low"),
                    ci_high=r.get("ci_high"),
                    source=r.get("source"),
                    elapsed_s=r.get("elapsed_s"),
                    n_epochs=r.get("n_epochs"),
                )
                for r in rows
            ]
        )

    @app.get("/api/v1/benchmarks/microqml/v1")
    def get_microqml_bench_v1() -> dict[str, Any]:
        """Versioned MicroQML Bench export (schema v1)."""
        return build_bench_export()

    return app


app = create_app()
