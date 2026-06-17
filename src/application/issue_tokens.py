"""Use case: issue JWT access and refresh tokens."""

from __future__ import annotations

from dataclasses import dataclass

from src.infrastructure.auth.jwt_service import (
    create_access_token,
    create_refresh_token_value,
    refresh_token_expiry,
    validate_api_key,
)
from src.infrastructure.database.repositories.sqlite_refresh_token_repository import (
    SqliteRefreshTokenRepository,
)
from src.shared.result import Result, fail, ok


@dataclass(frozen=True)
class IssueTokensDTO:
    tenant_id: str
    api_key: str
    user_id: str = "api-client"


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int = 900


class IssueTokensError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


def execute(
    dto: IssueTokensDTO,
    refresh_repo: SqliteRefreshTokenRepository,
) -> Result[TokenPair, IssueTokensError]:
    if not dto.tenant_id:
        return fail(IssueTokensError("INVALID_TENANT", "tenant_id is required"))
    if not validate_api_key(dto.api_key):
        return fail(IssueTokensError("INVALID_CREDENTIALS", "Invalid API key"))

    access_token = create_access_token(tenant_id=dto.tenant_id, user_id=dto.user_id)
    refresh_token = create_refresh_token_value()
    refresh_repo.save(
        tenant_id=dto.tenant_id,
        user_id=dto.user_id,
        token=refresh_token,
        expires_at=refresh_token_expiry(),
    )
    return ok(TokenPair(access_token=access_token, refresh_token=refresh_token))
