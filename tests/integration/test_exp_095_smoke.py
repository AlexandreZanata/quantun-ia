"""Smoke: exp_095 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_095_cybench_maize_slice.run import run_exp_095


def test_exp_095_import():
    assert callable(run_exp_095)
