"""Structured JSON logging for QML experiments (.cursor/rules/06-observability)."""

from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from loguru import logger

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_experiment_id: ContextVar[str | None] = ContextVar("experiment_id", default=None)
_seed: ContextVar[int | None] = ContextVar("seed", default=None)
_profile: ContextVar[str | None] = ContextVar("profile", default=None)

SERVICE_NAME = "quantun-ia"
SERVICE_VERSION = "0.4.0"


def init_correlation_id() -> str:
    correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    correlation_id = _correlation_id.get()
    if correlation_id is None:
        correlation_id = init_correlation_id()
    return correlation_id


def set_experiment_context(
    *,
    experiment_id: str | None = None,
    seed: int | None = None,
    profile: str | None = None,
) -> None:
    """Bind experiment metadata to the current async context for all subsequent log lines."""
    if experiment_id is not None:
        _experiment_id.set(experiment_id)
    if seed is not None:
        _seed.set(seed)
    if profile is not None:
        _profile.set(profile)


def log_event(level: str, msg: str, **context: Any) -> None:
    exp_id = context.get("exp_id") or context.get("experiment_id") or _experiment_id.get()
    seed = context.get("seed") if "seed" in context else _seed.get()
    profile = context.get("profile") if "profile" in context else _profile.get()

    record: dict[str, Any] = {
        "level": level,
        "time": datetime.now(timezone.utc).isoformat(),
        "correlationId": get_correlation_id(),
        "tenantId": "local",
        "userId": None,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "experimentId": exp_id,
        "seed": seed,
        "profile": profile,
        "msg": msg,
        "context": context,
    }
    line = json.dumps(record, default=str)
    if level == "error":
        logger.error(line)
    elif level == "warning":
        logger.warning(line)
    else:
        logger.info(line)
