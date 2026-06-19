"""Smoke test — exp_074 entanglement schedule NIHR imports."""

from experiments.exp_074_entangle_schedule_nihr.run import gate_passed, run_exp_074


def test_exp_074_import():
    assert callable(run_exp_074)
    assert callable(gate_passed)
