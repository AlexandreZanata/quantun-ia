"""Smoke: exp_084 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_084_residual_ft_nano_maize.run import run_exp_084


def test_exp_084_import():
    assert callable(run_exp_084)
