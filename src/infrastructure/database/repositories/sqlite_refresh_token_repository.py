"""SQLite persistence for refresh tokens."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class RefreshTokenRecord:
    id: str
    tenant_id: str
    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime
    revoked_at: datetime | None = None


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SqliteRefreshTokenRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, *, tenant_id: str, user_id: str, token: str, expires_at: datetime) -> str:
        token_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        self._conn.execute(
            """
            INSERT INTO refresh_tokens (
                id, tenant_id, user_id, token_hash, expires_at, revoked_at, created_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                token_id,
                tenant_id,
                user_id,
                _hash_token(token),
                expires_at.isoformat(),
                now.isoformat(),
            ),
        )
        self._conn.commit()
        return token_id

    def revoke(self, token: str) -> None:
        self._conn.execute(
            """
            UPDATE refresh_tokens
            SET revoked_at = ?
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (datetime.now(UTC).isoformat(), _hash_token(token)),
        )
        self._conn.commit()

    def find_active(self, token: str) -> RefreshTokenRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM refresh_tokens
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (_hash_token(token),),
        ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at <= datetime.now(UTC):
            return None
        return RefreshTokenRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            token_hash=row["token_hash"],
            expires_at=expires_at,
            created_at=datetime.fromisoformat(row["created_at"]),
            revoked_at=None,
        )
