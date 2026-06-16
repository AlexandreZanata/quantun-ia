"""Structured JSON logging for QML experiments (.cursor/rules/06-observability)."""

from __future__ import annotations

import json
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from loguru import logger

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

SERVICE_NAME = "quantun-ia"
SERVICE_VERSION = "0.1.0"


def init_correlation_id() -> str:
    correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    correlation_id = _correlation_id.get()
    if correlation_id is None:
        correlation_id = init_correlation_id()
    return correlation_id


def log_event(level: str, msg: str, **context) -> None:
    record = {
        "level": level,
        "time": datetime.now(timezone.utc).isoformat(),
        "correlationId": get_correlation_id(),
        "tenantId": "local",
        "userId": None,
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
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
