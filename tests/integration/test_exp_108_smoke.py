"""Smoke: exp_108 runner importable (no publication training)."""

from experiments.exp_108_quantum_flow_coupling.run import gate_passed, run_exp_108


def test_exp_108_import():
    assert callable(run_exp_108)
    assert callable(gate_passed)
