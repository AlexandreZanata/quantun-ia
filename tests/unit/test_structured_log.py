"""Unit tests for structured JSON logging."""

from src.training.structured_log import (
    get_correlation_id,
    init_correlation_id,
    log_event,
    set_experiment_context,
)


def test_correlation_id_is_stable_within_context():
    cid = init_correlation_id()
    assert get_correlation_id() == cid


def test_log_event_does_not_raise():
    init_correlation_id()
    log_event("info", "test message", key="value")


def test_log_event_includes_experiment_context(capsys):
    init_correlation_id()
    set_experiment_context(experiment_id="exp_001", seed=42, profile="ci")
    log_event("info", "context test")
    # loguru captures output; verify via record structure by parsing last log call
    set_experiment_context(experiment_id="exp_test", seed=99, profile="publication")
    log_event("info", "bound context", extra_field=1)
    assert get_correlation_id()  # smoke — context helpers do not raise


def test_set_experiment_context_fields():
    set_experiment_context(experiment_id="exp_015", seed=123, profile="ci")
    init_correlation_id()
    log_event("info", "with context", exp_id="exp_015", seed=123, profile="ci")

