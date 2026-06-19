"""Smoke test — exp_073 GoBug warm-start imports."""

from experiments.exp_073_quantum_warmstart_gobug.run import gate_passed, run_exp_073


def test_exp_073_import():
    assert callable(run_exp_073)
    assert callable(gate_passed)
