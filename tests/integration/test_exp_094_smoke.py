"""Smoke: exp_094 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_094_hard_temporal_drift.run import run_exp_094


def test_exp_094_import():
    assert callable(run_exp_094)
