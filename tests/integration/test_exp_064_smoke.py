"""Smoke test — exp_064 ACYD entanglement schedule imports."""

from experiments.exp_064_entangle_schedule_acyd.run import gate_passed, run_exp_064


def test_exp_064_import():
    assert callable(run_exp_064)
    assert callable(gate_passed)
