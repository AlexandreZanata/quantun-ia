"""Smoke test — exp_054 GV-ALR hybrid head imports."""

from __future__ import annotations

from experiments.exp_054_adaptive_hybrid_higgs.run import run_exp_054


def test_exp_054_import():
    assert callable(run_exp_054)
