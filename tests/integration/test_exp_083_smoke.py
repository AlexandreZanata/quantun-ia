"""Smoke test — exp_083 conventional ACYD maize baselines."""

from experiments.exp_083_conventional_acyd_maize_baselines.run import gate_passed, run_exp_083


def test_exp_083_import():
    assert callable(run_exp_083)
    assert callable(gate_passed)
