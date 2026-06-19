"""Smoke test — exp_061 conventional ACYD baselines."""

from experiments.exp_061_conventional_acyd_baselines.run import gate_passed, run_exp_061


def test_exp_061_import():
    assert callable(run_exp_061)
    assert callable(gate_passed)
