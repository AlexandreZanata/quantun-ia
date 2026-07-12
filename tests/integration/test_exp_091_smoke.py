"""Smoke: exp_091 runner is importable (no publication training)."""

from __future__ import annotations

from experiments.exp_091_circuit_cut_6q.run import run_exp_091


def test_exp_091_import():
    assert callable(run_exp_091)
