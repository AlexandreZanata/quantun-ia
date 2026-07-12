"""Smoke: exp_089 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_089_measurement_dropout_cal.run import run_exp_089


def test_exp_089_import():
    assert callable(run_exp_089)
