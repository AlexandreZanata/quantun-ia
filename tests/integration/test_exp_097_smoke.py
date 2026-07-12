"""Smoke: exp_097 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_097_spei_curriculum_agro.run import run_exp_097


def test_exp_097_import():
    assert callable(run_exp_097)
