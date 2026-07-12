"""Smoke test — exp_063 ACYD warm-start imports."""

from experiments.exp_063_quantum_warmstart_acyd.run import gate_passed, run_exp_063


def test_exp_063_import():
    assert callable(run_exp_063)
    assert callable(gate_passed)
