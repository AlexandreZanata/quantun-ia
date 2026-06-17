"""Use case: rotate refresh token and issue a new access token."""

from __future__ import annotations

from dataclasses import dataclass

from src.application.issue_tokens import TokenPair
from src.infrastructure.auth.jwt_service import (
    create_access_token,
    create_refresh_token_value,
    refresh_token_expiry,
)
from src.infrastructure.database.repositories.sqlite_refresh_token_repository import (
    SqliteRefreshTokenRepository,
)
from src.shared.result import Result, fail, ok


class RefreshTokensError:
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RefreshTokensDTO:
    refresh_token: str


def execute(
    dto: RefreshTokensDTO,
    refresh_repo: SqliteRefreshTokenRepository,
) -> Result[TokenPair, RefreshTokensError]:
    record = refresh_repo.find_active(dto.refresh_token)
    if record is None:
        return fail(RefreshTokensError("INVALID_REFRESH_TOKEN", "Refresh token invalid or expired"))

    refresh_repo.revoke(dto.refresh_token)
    new_refresh = create_refresh_token_value()
    refresh_repo.save(
        tenant_id=record.tenant_id,
        user_id=record.user_id,
        token=new_refresh,
        expires_at=refresh_token_expiry(),
    )
    access_token = create_access_token(tenant_id=record.tenant_id, user_id=record.user_id)
    return ok(TokenPair(access_token=access_token, refresh_token=new_refresh))
