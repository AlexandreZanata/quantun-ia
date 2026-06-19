"""Smoke test — exp_045 GoBug defect baseline imports."""

from __future__ import annotations

from experiments.exp_045_code_defect_gobug.run import run_exp_045


def test_exp_045_import():
    assert callable(run_exp_045)
