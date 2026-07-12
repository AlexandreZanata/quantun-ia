"""Smoke: exp_086 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_086_residual_qnn_head_maize.run import run_exp_086


def test_exp_086_import():
    assert callable(run_exp_086)
