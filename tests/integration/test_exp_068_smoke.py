"""Smoke test — exp_068 grand comparison imports."""

from experiments.exp_068_nano_grand_comparison.run import gate_passed, run_exp_068


def test_exp_068_import():
    assert callable(run_exp_068)
    assert callable(gate_passed)
