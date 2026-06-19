"""Smoke test — exp_068b compound stress ACYD imports."""

from experiments.exp_068b_compound_stress_acyd.run import gate_passed, run_exp_068b


def test_exp_068b_import():
    assert callable(run_exp_068b)
    assert callable(gate_passed)
