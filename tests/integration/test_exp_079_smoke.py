"""Smoke test — exp_079 quantum transfer imports."""

from experiments.exp_079_quantum_transfer_higgs_to_acyd.run import gate_passed, run_exp_079


def test_exp_079_import():
    assert callable(run_exp_079)
    assert callable(gate_passed)
