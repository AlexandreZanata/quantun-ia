"""Unit tests for pre-flight health checks."""

from scripts.health_check import run_health_check


def test_health_check_passes_in_repo():
    assert run_health_check(strict=True) == 0
