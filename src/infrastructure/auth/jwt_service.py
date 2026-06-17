"""JWT RS256 token issue and verification."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from src.infrastructure.auth.jwt_keys import load_public_key, load_signing_key

ACCESS_TOKEN_MINUTES = 15
REFRESH_TOKEN_DAYS = 7


@dataclass(frozen=True)
class AccessTokenClaims:
    tenant_id: str
    user_id: str
    subject: str


class JwtAuthError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(UTC)


def create_access_token(*, tenant_id: str, user_id: str) -> str:
    now = _now()
    payload = {
        "sub": user_id,
        "tenantId": tenant_id,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_MINUTES),
    }
    return jwt.encode(payload, load_signing_key(), algorithm="RS256")


def verify_access_token(token: str) -> AccessTokenClaims:
    try:
        payload = jwt.decode(
            token,
            load_public_key(),
            algorithms=["RS256"],
            options={"require": ["exp", "sub", "tenantId"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise JwtAuthError("TOKEN_EXPIRED", "Access token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise JwtAuthError("INVALID_TOKEN", "Invalid access token") from exc

    if payload.get("type") != "access":
        raise JwtAuthError("INVALID_TOKEN", "Token type must be access")

    tenant_id = str(payload["tenantId"])
    user_id = str(payload["sub"])
    if not tenant_id:
        raise JwtAuthError("INVALID_TENANT", "tenantId claim is required")
    return AccessTokenClaims(tenant_id=tenant_id, user_id=user_id, subject=user_id)


def create_refresh_token_value() -> str:
    return str(uuid.uuid4())


def refresh_token_expiry() -> datetime:
    return _now() + timedelta(days=REFRESH_TOKEN_DAYS)


def auth_required() -> bool:
    return os.environ.get("API_AUTH_REQUIRED", "0").strip().lower() in {"1", "true", "yes"}


def validate_api_key(api_key: str) -> bool:
    expected = os.environ.get("API_AUTH_SECRET", "dev-secret-change-me")
    return api_key == expected
