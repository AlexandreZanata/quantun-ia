"""FastAPI dependencies for tenant auth context."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from src.infrastructure.auth.jwt_service import (
    AccessTokenClaims,
    JwtAuthError,
    auth_required,
    verify_access_token,
)


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    user_id: str | None = None


def resolve_tenant_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> TenantContext:
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        try:
            claims: AccessTokenClaims = verify_access_token(token)
        except JwtAuthError as exc:
            raise HTTPException(
                status_code=401,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        return TenantContext(tenant_id=claims.tenant_id, user_id=claims.user_id)

    if auth_required():
        raise HTTPException(
            status_code=401,
            detail={"code": "UNAUTHORIZED", "message": "Bearer token required"},
        )

    tenant_id = x_tenant_id or "local"
    return TenantContext(tenant_id=tenant_id, user_id=None)
