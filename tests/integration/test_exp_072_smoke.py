"""Smoke test — exp_072 NIHR warm-start imports."""

from experiments.exp_072_quantum_warmstart_nihr.run import gate_passed, run_exp_072


def test_exp_072_import():
    assert callable(run_exp_072)
    assert callable(gate_passed)
