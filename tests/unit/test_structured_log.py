"""Unit tests for structured JSON logging."""

from src.training.structured_log import get_correlation_id, init_correlation_id, log_event


def test_correlation_id_is_stable_within_context():
    cid = init_correlation_id()
    assert get_correlation_id() == cid


def test_log_event_does_not_raise():
    init_correlation_id()
    log_event("info", "test message", key="value")
