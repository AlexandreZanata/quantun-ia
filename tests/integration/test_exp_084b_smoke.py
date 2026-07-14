"""Smoke: exp_084b runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_084b_residual_nano_soy_transfer.run import run_exp_084b


def test_exp_084b_import():
    assert callable(run_exp_084b)
