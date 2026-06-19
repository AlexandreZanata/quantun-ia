"""Smoke test — exp_051 quantum head NIHR imports."""

from __future__ import annotations

from experiments.exp_051_quantum_head_nihr.run import run_exp_051


def test_exp_051_import():
    assert callable(run_exp_051)
