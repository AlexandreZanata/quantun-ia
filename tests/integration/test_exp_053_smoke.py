"""Smoke test — exp_053 entanglement schedule imports."""

from experiments.exp_053_entangle_schedule_bc.run import gate_passed, run_exp_053


def test_exp_053_import():
    assert callable(run_exp_053)
    assert callable(gate_passed)
