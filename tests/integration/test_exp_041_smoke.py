"""Smoke test — exp_041 clinical validation imports."""

from __future__ import annotations

from experiments.exp_041_human_cv_clinical_cases.run import run_exp_041


def test_exp_041_import():
    assert callable(run_exp_041)
