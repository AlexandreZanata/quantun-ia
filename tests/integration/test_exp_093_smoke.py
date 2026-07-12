"""Smoke: exp_093 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_093_pqk_ridge_head.run import run_exp_093


def test_exp_093_import():
    assert callable(run_exp_093)
