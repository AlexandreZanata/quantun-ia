"""Smoke test — exp_052 quantum warm-start imports."""

from __future__ import annotations

from experiments.exp_052_quantum_warmstart_higgs.run import gate_passed, run_exp_052


def test_exp_052_import():
    assert callable(run_exp_052)
    assert callable(gate_passed)
