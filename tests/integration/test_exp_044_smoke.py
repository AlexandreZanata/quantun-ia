"""Smoke test — exp_044 NIHR CV baseline imports."""

from __future__ import annotations

from experiments.exp_044_nihr_cv_baseline.run import run_exp_044


def test_exp_044_import():
    assert callable(run_exp_044)
